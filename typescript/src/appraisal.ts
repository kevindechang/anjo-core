/**
 * Non-habituating OCC/PAD appraisal as pure state transforms.
 *
 * The AR(1) home-base-plus-attractor form follows DynAffect (Kuppens, Oravecz &
 * Tuerlinckx 2010); the emotion names are an OCC subset (Ortony, Clore &
 * Collins 1988), but this is a lookup table keyed on a pre-classified intent,
 * not an appraisal process over OCC appraisal variables. Constant provenance —
 * literature, production-tuned, or bounded arbitrary — is recorded per constant
 * in docs/foundations.md sections 2-5.
 */
import type { AppraisalGoals, PadMood, Personality } from './contracts.js';
import { DEFAULT_APPRAISAL_GOALS, DEFAULT_PERSONALITY } from './contracts.js';
import { pyRound } from './internal/round.js';

export const DEFAULT_STAGES: readonly string[] = Object.freeze([
  'stranger',
  'acquaintance',
  'friend',
  'close',
  'intimate',
]);
export const DEFAULT_STAGE_WEIGHTS: readonly number[] = Object.freeze([0, 0.2, 0.4, 0.6, 0.7]);

/** Raised when a strict {@link StageLadder} receives a stage it does not define. */
export class UnknownStageError extends Error {
  constructor(stage: string, stages: readonly string[]) {
    super(`stage ${JSON.stringify(stage)} is not on the ladder [${stages.join(', ')}]`);
    this.name = 'UnknownStageError';
  }
}

/**
 * An ordered relationship ladder and the resting-point weight of each rung.
 *
 * The default rungs are the reference conversational ladder. Domains with a
 * different progression -- a game faction track, a tutoring competence ladder,
 * a support-case lifecycle -- should supply their own stages and weights
 * rather than reusing these labels.
 *
 * `strict` controls unknown stages. The default (`false`) floors to the first
 * rung, matching the pinned cross-runtime contract; `true` throws
 * {@link UnknownStageError} instead, so a typo cannot silently resolve to
 * weight `0` and remove the resting set point.
 */
export interface StageLadder {
  readonly stages: readonly string[];
  readonly weights: readonly number[];
  readonly strict: boolean;
}

export const DEFAULT_STAGE_LADDER: StageLadder = Object.freeze({
  stages: DEFAULT_STAGES,
  weights: DEFAULT_STAGE_WEIGHTS,
  strict: false,
});

/** Validate and freeze a ladder; mirrors the Python `StageLadder` constructor. */
export function createStageLadder(input: Partial<StageLadder> = {}): StageLadder {
  const stages = Object.freeze([...(input.stages ?? DEFAULT_STAGES)]);
  const weights = Object.freeze([...(input.weights ?? DEFAULT_STAGE_WEIGHTS)]);
  const strict = input.strict ?? false;
  if (stages.length === 0) throw new RangeError('stages must not be empty');
  if (stages.length !== weights.length) {
    throw new RangeError('stages and weights must have equal length');
  }
  if (new Set(stages).size !== stages.length) throw new RangeError('stages must be unique');
  for (const stage of stages) {
    if (typeof stage !== 'string' || stage.length === 0) {
      throw new TypeError('stage names must be non-empty strings');
    }
  }
  for (const weight of weights) {
    if (!Number.isFinite(weight) || weight < 0 || weight > 1) {
      throw new RangeError('stage weights must be finite and within [0, 1]');
    }
  }
  return Object.freeze({ stages, weights, strict });
}

/** Whether `stage` is a defined rung, for validation at an application boundary. */
export function ladderKnows(ladder: StageLadder, stage: string): boolean {
  return ladder.stages.indexOf(stage) !== -1;
}

const AMBIGUOUS_INTENTS = new Set(['CASUAL', 'CURIOSITY', 'CHALLENGE', 'APOLOGY']);
const OCC_DECAY: Readonly<Record<string, number>> = Object.freeze({
  reproach: 0.7,
  distress: 0.8,
  admiration: 0.85,
  gratitude: 0.88,
  joy: 0.9,
});

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export { DEFAULT_APPRAISAL_GOALS, DEFAULT_PERSONALITY };

export function stageInt(
  stage: string | null | undefined,
  ladder: StageLadder = DEFAULT_STAGE_LADDER,
): number {
  const index = stage === null || stage === undefined ? -1 : ladder.stages.indexOf(stage);
  if (index !== -1) return index + 1;
  if (ladder.strict) throw new UnknownStageError(String(stage), ladder.stages);
  return 1;
}

export function moodInertia(personality: Personality): number {
  const inertia = 0.8 + 0.2 * (personality.N - 0.5) - 0.1 * (personality.E - 0.5);
  return clamp(inertia, 0.62, 0.92);
}

export function baselineWeight(
  stage: number,
  ladder: StageLadder = DEFAULT_STAGE_LADDER,
): number {
  if (!Number.isInteger(stage)) throw new TypeError('stage ordinal must be an integer');
  if (stage >= 1 && stage <= ladder.weights.length) return ladder.weights[stage - 1] as number;
  return 0;
}

export function decayMood(
  mood: PadMood,
  personality: Personality,
  stage: number,
  baselineValence: number,
  ladder: StageLadder = DEFAULT_STAGE_LADDER,
): PadMood {
  const inertia = moodInertia(personality);
  const weight = baselineWeight(stage, ladder);
  const setValence = weight * baselineValence;
  const setDominance = weight * 0.1;
  return {
    valence: pyRound(clamp(setValence + inertia * (mood.valence - setValence), -1, 1), 4),
    arousal: pyRound(clamp(inertia * mood.arousal, -1, 1), 4),
    dominance: pyRound(
      clamp(setDominance + inertia * (mood.dominance - setDominance), -1, 1),
      4,
    ),
  };
}

export interface AppraiseInputResult {
  emotions: Record<string, number>;
  mood: PadMood;
  baselineValence: number;
}

export function appraiseInput(
  mood: PadMood,
  goals: AppraisalGoals,
  intent: string,
  baselineValence: number,
): AppraiseInputResult {
  const emotions: Record<string, number> = {
    joy: 0,
    distress: 0,
    admiration: 0,
    reproach: 0,
    gratitude: 0,
  };
  let { valence, arousal, dominance } = mood;

  switch (intent) {
    case 'ABUSE':
      dominance = Math.min(1, dominance + 0.25);
      valence = Math.max(-1, valence - 0.35);
      arousal = Math.max(-1, arousal - 0.1);
      emotions.reproach = Math.min(1, goals.respect * 0.95);
      emotions.distress = Math.min(1, goals.rapport * 0.65);
      break;
    case 'APOLOGY':
      valence = Math.min(1, valence + 0.05);
      emotions.joy = Math.min(1, goals.rapport * 0.2);
      emotions.gratitude = Math.min(1, goals.honesty * 0.35);
      break;
    case 'VULNERABILITY':
      valence = Math.min(1, valence + 0.15);
      arousal = Math.min(1, arousal + 0.1);
      emotions.gratitude = Math.min(1, goals.honesty * 0.7);
      emotions.joy = Math.min(1, goals.rapport * 0.75);
      break;
    case 'CURIOSITY':
      valence = Math.min(1, valence + 0.2);
      arousal = Math.min(1, arousal + 0.15);
      dominance = Math.min(1, dominance + 0.05);
      emotions.admiration = Math.min(1, goals.intellectual * 0.75);
      emotions.joy = Math.min(1, goals.rapport * 0.5);
      break;
    case 'CHALLENGE':
      dominance = Math.min(1, dominance + 0.1);
      valence = Math.max(-1, valence - 0.05);
      emotions.admiration = Math.min(1, goals.intellectual * 0.45);
      emotions.distress = Math.min(1, goals.rapport * 0.25);
      break;
    case 'NEGLECT':
      valence = Math.max(-1, valence - 0.1);
      arousal = Math.max(-1, arousal - 0.05);
      emotions.distress = Math.min(1, goals.rapport * 0.4);
      break;
    case 'CASUAL':
      valence = Math.min(1, valence + 0.02);
      emotions.joy = 0.05;
      break;
    default:
      break;
  }

  if (AMBIGUOUS_INTENTS.has(intent) && Math.abs(valence) >= 0.2) {
    const gain = valence < 0 ? 0.1 : 0.04;
    valence = pyRound(clamp(valence * (1 + gain), -1, 1), 4);
  }
  const nextBaseline = pyRound(clamp(baselineValence * 0.98 + valence * 0.02, -1, 1), 4);
  return {
    emotions,
    mood: { valence, arousal, dominance },
    baselineValence: nextBaseline,
  };
}

const WORD_CHARACTER = /[\p{L}\p{N}_]/u;

function isWordCharacter(character: string | undefined): boolean {
  return character !== undefined && WORD_CHARACTER.test(character);
}

function wordMatch(text: string, tokens: readonly string[]): boolean {
  for (const raw of tokens) {
    const stem = raw.endsWith('-');
    const needle = stem ? raw.slice(0, -1) : raw;
    if (!needle) continue;
    for (let index = text.indexOf(needle); index !== -1; index = text.indexOf(needle, index + 1)) {
      const startsAtBoundary = !isWordCharacter(text[index - 1]);
      const endsAtBoundary = stem || !isWordCharacter(text[index + needle.length]);
      if (startsAtBoundary && endsAtBoundary) return true;
    }
  }
  return false;
}

/**
 * The lexical cues used to detect a violated expectation.
 *
 * These defaults are an **English conversational preset**, not domain-neutral
 * behavior. Supply your own tokens to localize the kernel or to express
 * expectation violation in a non-conversational vocabulary. A trailing `-`
 * marks a stem match (`'apolog-'` matches `apologize` and `apology`).
 */
export interface ExpectationCues {
  readonly negativeExpected: readonly string[];
  readonly positiveCurrent: readonly string[];
  readonly negativeCurrent: readonly string[];
  readonly expectedResolution: readonly string[];
}

export const DEFAULT_EXPECTATION_CUES: ExpectationCues = Object.freeze({
  negativeExpected: Object.freeze([
    'angry', 'upset', 'bad', 'worse', 'argument', 'hurt', 'afraid', 'scared',
  ]),
  positiveCurrent: Object.freeze([
    'better', 'okay', 'ok', 'good', 'fine', 'resolved', 'worked out',
  ]),
  negativeCurrent: Object.freeze([
    'worse', 'bad', 'angry', 'upset', 'failed', 'awful', 'hurt',
  ]),
  expectedResolution: Object.freeze(['come back', 'tell', 'answer', 'resolve', 'apolog-']),
});

export function expectationEmotions(
  expectation: string | null | undefined,
  message: string | null | undefined,
  cues: ExpectationCues = DEFAULT_EXPECTATION_CUES,
): Record<string, number> {
  const expected = (expectation ?? '').toLowerCase();
  const current = (message ?? '').toLowerCase();
  if (!expected || !current) return {};

  const negativeExpected = wordMatch(expected, cues.negativeExpected);
  const positiveCurrent = wordMatch(current, cues.positiveCurrent);
  const negativeCurrent = wordMatch(current, cues.negativeCurrent);
  const expectedResolution = wordMatch(expected, cues.expectedResolution);

  if (negativeExpected && positiveCurrent) return { relief: 0.45, surprise: 0.35 };
  if (expectedResolution && negativeCurrent) return { disappointment: 0.4 };
  if (expectedResolution && positiveCurrent) return { satisfaction: 0.4 };
  return {};
}

export function occCarryDecay(carry: Readonly<Record<string, number>>): Record<string, number> {
  const result: Record<string, number> = {};
  for (const [emotion, intensity] of Object.entries(carry)) {
    const decayed = intensity * (OCC_DECAY[emotion] ?? 0.8);
    if (decayed > 0.05) result[emotion] = decayed;
  }
  return result;
}

/** Name aligned with the Python package. */
export const decayOccCarry = occCarryDecay;

export function stateEmotions(mood: PadMood, attachmentLonging: number): Record<string, number> {
  const result: Record<string, number> = {};
  if (mood.arousal < 0) result.fatigue = pyRound(Math.min(1, -mood.arousal), 3);
  if (attachmentLonging > 0.3) result.longing = pyRound(attachmentLonging, 3);
  if (mood.valence > -0.3 && mood.valence < 0) {
    result.unease = pyRound(Math.min(0.3, -mood.valence), 3);
  }
  return result;
}

function mergeMax(
  base: Readonly<Record<string, number>>,
  extra: Readonly<Record<string, number>>,
): Record<string, number> {
  const result = { ...base };
  for (const [emotion, intensity] of Object.entries(extra)) {
    if (intensity > (result[emotion] ?? 0)) result[emotion] = intensity;
  }
  return result;
}

export interface AppraiseTurnInput {
  mood: PadMood;
  personality: Personality;
  goals: AppraisalGoals;
  stageInt: number;
  baselineValence: number;
  attachmentLonging: number;
  intent: string;
  occCarry?: Readonly<Record<string, number>>;
  expectation?: string | null;
  message?: string | null;
  /** Ladder used for the resting-point weight; defaults to the conversational rungs. */
  ladder?: StageLadder;
  /** Expectation-violation vocabulary; defaults to the English conversational preset. */
  cues?: ExpectationCues;
}

export interface AppraiseTurnResult {
  mood: PadMood;
  activeEmotions: Record<string, number>;
  occCarry: Record<string, number>;
  baselineValence: number;
}

/**
 * Non-habituating per-turn OCC/PAD transform. Intent classification is an
 * input; this function never invokes a model or persistence layer.
 */
export function appraiseTurn(input: AppraiseTurnInput): AppraiseTurnResult {
  const decayedMood = decayMood(
    input.mood,
    input.personality,
    input.stageInt,
    input.baselineValence,
    input.ladder ?? DEFAULT_STAGE_LADDER,
  );
  const appraisal = appraiseInput(
    decayedMood,
    input.goals,
    input.intent,
    input.baselineValence,
  );
  const decayedCarry = occCarryDecay(input.occCarry ?? {});
  const keys = [...new Set([
    ...Object.keys(appraisal.emotions),
    ...Object.keys(decayedCarry),
  ])].sort();
  let active: Record<string, number> = {};
  for (const key of keys) {
    active[key] = Math.max(appraisal.emotions[key] ?? 0, decayedCarry[key] ?? 0);
  }
  active = mergeMax(active, stateEmotions(appraisal.mood, input.attachmentLonging));
  active = mergeMax(
    active,
    expectationEmotions(input.expectation, input.message, input.cues ?? DEFAULT_EXPECTATION_CUES),
  );

  const carry: Record<string, number> = {};
  for (const [emotion, intensity] of Object.entries(active)) {
    if (intensity > 0.05) carry[emotion] = intensity;
  }
  return {
    mood: appraisal.mood,
    activeEmotions: active,
    occCarry: carry,
    baselineValence: appraisal.baselineValence,
  };
}
