import type { AffectState } from './contracts.js';
import { createAffectState } from './contracts.js';
import { pyRound } from './internal/round.js';
import { rstripPyWhitespace, splitPyWhitespace, stripPyWhitespace } from './internal/whitespace.js';

function smartTrim(text: string, maxChars: number): string {
  const cleaned = splitPyWhitespace(text).join(' ');
  if (maxChars <= 0 || cleaned.length <= maxChars) return cleaned;
  const window = cleaned.slice(0, maxChars);
  const sentenceEnd = Math.max(window.lastIndexOf('.'), window.lastIndexOf('!'), window.lastIndexOf('?'));
  if (sentenceEnd >= maxChars * 0.5) return rstripPyWhitespace(window.slice(0, sentenceEnd + 1));
  const lastSpace = window.lastIndexOf(' ');
  if (lastSpace === -1) return rstripPyWhitespace(window);
  const trimmed = rstripPyWhitespace(window.slice(0, lastSpace)).replace(/[,;:—-]+$/u, '');
  return trimmed ? `${trimmed}…` : rstripPyWhitespace(window);
}

export function cleanText(value: unknown, maxLength: number): string | null {
  if (typeof value !== 'string') return null;
  const unquoted = stripPyWhitespace(value).replace(/^"+|"+$/g, '');
  const cleaned = splitPyWhitespace(unquoted).join(' ');
  return cleaned ? smartTrim(cleaned, maxLength) : null;
}

export interface PresenceLineInput {
  reflectionPending: boolean;
  dueIntention: boolean;
  carriedThought: boolean;
  openThread: boolean;
}

/**
 * The rendered wording of the presence surface.
 *
 * The defaults are the reference conversational phrasing. They are English and
 * companion-shaped on purpose; a game, tutoring, or support domain should pass
 * its own labels rather than surfacing "here with you" to its users.
 */
export interface PresenceLabels {
  readonly reflecting: string;
  readonly dueIntention: string;
  readonly carriedThought: string;
  readonly openThread: string;
  readonly idle: string;
  readonly reflectingMode: string;
  readonly idleMode: string;
}

export const DEFAULT_PRESENCE_LABELS: PresenceLabels = Object.freeze({
  reflecting: 'still reflecting',
  dueIntention: 'holding a follow-up',
  carriedThought: 'carrying a thread',
  openThread: 'holding a pattern',
  idle: 'here with you',
  reflectingMode: 'reflecting',
  idleMode: 'quiet',
});

export function presenceLine(
  input: PresenceLineInput,
  labels: PresenceLabels = DEFAULT_PRESENCE_LABELS,
): string {
  if (input.reflectionPending) return labels.reflecting;
  if (input.dueIntention) return labels.dueIntention;
  if (input.carriedThought) return labels.carriedThought;
  if (input.openThread) return labels.openThread;
  return labels.idle;
}

export interface CognitionState {
  reflectionPending?: boolean;
  carriedThought?: boolean;
  dueIntention?: boolean;
  openThread?: boolean;
  intentionality?: boolean;
  curiosity?: boolean;
}

export interface PresenceVector {
  trust: number;
  valence: number;
  arousal: number;
  longing: number;
  awaiting: boolean;
  mode: string;
  line: string;
  affect: { valence: number; arousal: number; dominance: number };
  relationship: { stage: string; trust: number; longing: number; comfort: number };
  cognition: {
    reflection_pending: boolean;
    carried_thought: boolean;
    due_intention: boolean;
    open_thread: boolean;
    intentionality: boolean;
    curiosity: boolean;
  };
  source: 'affect_state';
}

export function buildPresenceVector(
  stateInput: AffectState,
  cognitionInput: CognitionState = {},
  labels: PresenceLabels = DEFAULT_PRESENCE_LABELS,
): PresenceVector {
  const state = createAffectState(stateInput);
  const cognition: Required<CognitionState> = {
    reflectionPending: cognitionInput.reflectionPending ?? false,
    carriedThought: Boolean(cleanText(state.carriedThought, 300)),
    dueIntention: cognitionInput.dueIntention ?? false,
    openThread: cognitionInput.openThread ?? false,
    intentionality: cognitionInput.intentionality ?? false,
    curiosity: cognitionInput.curiosity ?? false,
  };
  const trust = pyRound(state.relationship.trustScore, 4);
  const valence = pyRound(state.mood.valence, 4);
  const arousal = pyRound(state.mood.arousal, 4);
  const longing = pyRound(state.attachment.longing, 4);
  return {
    trust,
    valence,
    arousal,
    longing,
    awaiting: cognition.reflectionPending,
    mode: cognition.reflectionPending ? labels.reflectingMode : labels.idleMode,
    line: presenceLine(cognition, labels),
    affect: { valence, arousal, dominance: pyRound(state.mood.dominance, 4) },
    relationship: {
      stage: state.relationship.stage,
      trust,
      longing,
      comfort: pyRound(state.attachment.comfort, 4),
    },
    cognition: {
      reflection_pending: cognition.reflectionPending,
      carried_thought: cognition.carriedThought,
      due_intention: cognition.dueIntention,
      open_thread: cognition.openThread,
      intentionality: cognition.intentionality,
      curiosity: cognition.curiosity,
    },
    source: 'affect_state',
  };
}
