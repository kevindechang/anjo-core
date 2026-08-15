import type { DecodingParams, Message, PadMood } from './contracts.js';
import { pyRound } from './internal/round.js';
import { rstripPyWhitespace } from './internal/whitespace.js';

const OCTANTS: Readonly<Record<string, string>> = Object.freeze({
  '1,1,1': 'exuberant',
  '1,1,-1': 'dependent',
  '1,-1,1': 'relaxed',
  '1,-1,-1': 'docile',
  '-1,1,1': 'hostile',
  '-1,1,-1': 'anxious',
  '-1,-1,1': 'disdainful',
  '-1,-1,-1': 'bored',
});

const POSITIVE_EMOTIONS = new Set([
  'joy', 'admiration', 'gratitude', 'relief', 'satisfaction', 'hope',
]);
const NEGATIVE_EMOTIONS = new Set([
  'distress', 'reproach', 'fear', 'disappointment', 'unease', 'anger', 'sadness',
]);

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function sign(value: number): number {
  return value >= 0 ? 1 : -1;
}

/** Return the ALMA PAD octant, with a deadband around neutral. */
export function moodOctant(valence: number, arousal: number, dominance: number): string {
  if (Math.abs(valence) < 0.15 && Math.abs(arousal) < 0.15 && Math.abs(dominance) < 0.15) {
    return 'neutral';
  }
  return OCTANTS[`${sign(valence)},${sign(arousal)},${sign(dominance)}`] as string;
}

/** Map arousal to a bounded sampling envelope. */
export function decodingParams(mood: PadMood | null): DecodingParams {
  if (mood === null) return { temperature: 1, topP: null };
  return {
    temperature: clamp(1 + 0.2 * mood.arousal, 0.72, 1.18),
    topP: 0.97,
  };
}

/** Response ceiling multiplier; the deterministic control only shortens. */
export function lengthFactor(mood: PadMood | null): number {
  if (mood === null) return 1;
  if (mood.arousal < -0.3) return 0.6;
  if (mood.valence < -0.3) return 0.72;
  return 1;
}

export function applyLengthFactor(baseTokens: number, mood: PadMood | null): number {
  if (!Number.isSafeInteger(baseTokens) || baseTokens <= 0) {
    throw new RangeError('baseTokens must be a positive integer');
  }
  const factor = lengthFactor(mood);
  if (factor >= 1) return baseTokens;
  return Math.min(baseTokens, Math.max(180, pyRound(baseTokens * factor, 0)));
}

/** Whether opposing emotion families are both materially active. */
export function isAmbivalent(emotions: Readonly<Record<string, number>> | null | undefined): boolean {
  if (!emotions) return false;
  let positive = 0;
  let negative = 0;
  for (const [name, intensity] of Object.entries(emotions)) {
    if (POSITIVE_EMOTIONS.has(name)) positive = Math.max(positive, intensity);
    if (NEGATIVE_EMOTIONS.has(name)) negative = Math.max(negative, intensity);
  }
  if (positive < 0.15 || negative < 0.15) return false;
  return Math.min(positive, negative) / Math.max(positive, negative) >= 0.4;
}

/** All prose emitted by turn shaping is caller-owned through this policy. */
export interface TurnShapePolicy {
  heading: string;
  baseRules: ReadonlyArray<string>;
  afterQuestionRule: string;
  defaultCloseRule: string;
  ambivalenceRule: string;
  moodCues: Readonly<Partial<Record<string, string>>>;
  /**
   * Intent (upper-case) to the mood cues it suppresses. A policy that omits
   * this suppresses nothing; the reference rule lives in
   * {@link NEUTRAL_TURN_SHAPE_POLICY}, not in the kernel.
   */
  suppressedMoodCues?: Readonly<Partial<Record<string, readonly string[]>>>;
}

/** Neutral synthetic wording; applications should supply domain-specific policy. */
export const NEUTRAL_TURN_SHAPE_POLICY: Readonly<TurnShapePolicy> = Object.freeze({
  heading: 'Response shape',
  baseRules: Object.freeze([
    'Keep one main idea in focus.',
    "Keep the response proportionate to the user's message.",
  ]),
  afterQuestionRule: 'Prefer a statement ending after a previous question.',
  defaultCloseRule: 'Use a question only when it materially helps the exchange.',
  ambivalenceRule: 'Preserve meaningful opposing signals instead of forcing a verdict.',
  moodCues: Object.freeze({
    exuberant: 'Allow somewhat more momentum.',
    dependent: 'Keep a warm, compact cadence.',
    relaxed: 'Use an easy, unhurried pace.',
    docile: 'Keep the response quiet and spare.',
    hostile: 'Use a composed, plain register.',
    anxious: 'Keep the response shorter and less resolved.',
    disdainful: 'Use a cool, economical register.',
    bored: 'Keep the response minimal and unforced.',
  }),
  // Reference conversational rule expressed as data: do not add upbeat momentum
  // right after a user has been vulnerable. Override or drop it freely.
  suppressedMoodCues: Object.freeze({ VULNERABILITY: Object.freeze(['exuberant']) }),
});

export interface TurnShapeContext {
  mood: PadMood | null;
  intent?: string;
  history?: ReadonlyArray<Message | { role: string; content: unknown }> | null;
  emotions?: Readonly<Record<string, number>> | null;
}

/**
 * Resolve the suppressed cues for an intent, matching declared keys
 * case-insensitively. Python normalizes declared keys when the policy is
 * constructed; TypeScript policies are plain object literals with no
 * construction step, so the normalization has to happen at lookup or the two
 * runtimes disagree on a lower-case declaration.
 */
function suppressedFor(policy: TurnShapePolicy, intent: string | undefined): readonly string[] {
  const declared = policy.suppressedMoodCues;
  if (!declared || intent === undefined) return [];
  const wanted = intent.toUpperCase();
  const direct = declared[wanted];
  if (direct) return direct;
  for (const [key, octants] of Object.entries(declared)) {
    if (key.toUpperCase() === wanted && octants) return octants;
  }
  return [];
}

function lastAssistantText(history: TurnShapeContext['history']): string {
  if (!history) return '';
  for (let index = history.length - 1; index >= 0; index -= 1) {
    const message = history[index];
    if (message?.role === 'assistant') {
      return typeof message.content === 'string' ? message.content : '';
    }
  }
  return '';
}

/** Select structural cues while leaving every word caller-configurable. */
export function buildTurnShapeDirective(
  context: TurnShapeContext,
  policy: TurnShapePolicy = NEUTRAL_TURN_SHAPE_POLICY,
): string {
  const rules = [...policy.baseRules];
  const previousAsked = rstripPyWhitespace(lastAssistantText(context.history)).endsWith('?');
  const closeRule = previousAsked ? policy.afterQuestionRule : policy.defaultCloseRule;
  if (closeRule) rules.push(closeRule);

  if (context.mood !== null) {
    const octant = moodOctant(
      context.mood.valence,
      context.mood.arousal,
      context.mood.dominance,
    );
    const suppressed = suppressedFor(policy, context.intent);
    const cue = suppressed.includes(octant) ? undefined : policy.moodCues[octant];
    if (cue) rules.push(cue);
  }
  if (isAmbivalent(context.emotions) && policy.ambivalenceRule) {
    rules.push(policy.ambivalenceRule);
  }

  const nonempty = rules.filter((rule) => rule.length > 0);
  if (!policy.heading && nonempty.length === 0) return '';
  const lines = policy.heading ? [`${policy.heading}:`] : [];
  lines.push(...nonempty.map((rule) => `- ${rule}`));
  return lines.join('\n');
}

export interface TurnShapeOptions extends Omit<TurnShapeContext, 'mood'> {
  policy?: TurnShapePolicy;
}

/** Signature-compatible convenience wrapper around buildTurnShapeDirective. */
export function turnShapeDirective(mood: PadMood | null, options: TurnShapeOptions = {}): string {
  return buildTurnShapeDirective(
    {
      mood,
      ...(options.intent === undefined ? {} : { intent: options.intent }),
      ...(options.history === undefined ? {} : { history: options.history }),
      ...(options.emotions === undefined ? {} : { emotions: options.emotions }),
    },
    options.policy ?? NEUTRAL_TURN_SHAPE_POLICY,
  );
}
