import type { CompanionState } from './contracts.js';
import { createCompanionState } from './contracts.js';
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

export function presenceLine(input: PresenceLineInput): string {
  if (input.reflectionPending) return 'still reflecting';
  if (input.dueIntention) return 'holding a follow-up';
  if (input.carriedThought) return 'carrying a thread';
  if (input.openThread) return 'holding a pattern';
  return 'here with you';
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
  source: 'companion_state';
}

export function buildPresenceVector(
  stateInput: CompanionState,
  cognitionInput: CognitionState = {},
): PresenceVector {
  const state = createCompanionState(stateInput);
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
    mode: cognition.reflectionPending ? 'reflecting' : 'quiet',
    line: presenceLine(cognition),
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
    source: 'companion_state',
  };
}
