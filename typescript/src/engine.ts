import { buildTurnShapeDirective, decodingParams } from './affect.js';
import type { TurnShapePolicy } from './affect.js';
import { appraiseTurn, stageInt } from './appraisal.js';
import type { AppraiseTurnInput, AppraiseTurnResult } from './appraisal.js';
import type {
  AdapterControl,
  AbortSignalLike,
  AffectState,
  DeepReadonly,
  GateResult,
  Intent,
  MemoryCandidate,
  MemoryRetriever,
  Message,
  ModelAdapter,
  RankedMemory,
  StateStore,
  StateTransaction,
  TurnCallbacks,
  TurnResult,
} from './contracts.js';
import { createAffectState, INTENTS } from './contracts.js';
import { readonlySnapshot } from './internal/snapshot.js';
import { stripPyWhitespace } from './internal/whitespace.js';
import { buildUntrustedContext, composePrompt } from './prompt.js';
import type { PromptPolicy } from './prompt.js';
import { rankCandidates } from './retrieval.js';
import { DEFAULT_PRESENCE_LABELS, buildPresenceVector, type PresenceLabels } from './surfacing.js';
import type { CognitionState, PresenceVector } from './surfacing.js';

const INTENT_SET: ReadonlySet<string> = new Set(INTENTS);
const CUSTOM_INTENT = /^[A-Z][A-Z0-9_]{0,63}$/u;

export interface EngineLimits {
  readonly maxMessageChars: number;
  readonly maxHistoryMessages: number;
  readonly maxHistoryChars: number;
  readonly maxCandidates: number;
  readonly maxMemories: number;
  readonly maxMemoryChars: number;
  readonly maxPromptChars: number;
  readonly maxOutputChars: number;
  /** Maximum turns admitted by one engine instance, including the active turn. */
  readonly maxQueuedTurns: number;
}

export const DEFAULT_ENGINE_LIMITS: Readonly<EngineLimits> = Object.freeze({
  maxMessageChars: 16_000,
  maxHistoryMessages: 200,
  maxHistoryChars: 128_000,
  maxCandidates: 100,
  maxMemories: 20,
  maxMemoryChars: 8_000,
  maxPromptChars: 128_000,
  maxOutputChars: 32_000,
  maxQueuedTurns: 32,
});

export type AppraisalPolicy = (
  input: Readonly<AppraiseTurnInput>,
) => AppraiseTurnResult;

/** Current conversational OCC/PAD mapping, exposed as the reference policy. */
export const DEFAULT_APPRAISAL_POLICY: AppraisalPolicy = (input) => appraiseTurn(input);

export interface AffectEngineOptions {
  readonly model: ModelAdapter;
  readonly store: StateStore;
  readonly retriever?: MemoryRetriever;
  readonly instruction?: string;
  readonly promptPolicy?: PromptPolicy;
  readonly turnShapePolicy?: TurnShapePolicy;
  /** Wording of the presence surface; defaults to the conversational phrasing. */
  readonly presenceLabels?: PresenceLabels;
  readonly stateFactory?: () => AffectState;
  readonly retrievalLimit?: number;
  /** Opt-in fallback; omitted means gate errors and malformed output propagate. */
  readonly gateFallback?: GateResult;
  /** Additional normalized labels; custom labels intentionally have no built-in appraisal impulse. */
  readonly customIntents?: Iterable<string>;
  readonly appraisalPolicy?: AppraisalPolicy;
  readonly limits?: Partial<EngineLimits>;
  readonly now?: () => Date;
}

function parseIntent(value: unknown, allowed: ReadonlySet<string>): Intent {
  if (typeof value !== 'string') throw new TypeError('gate intent must be a string');
  const normalized = value.trim().toUpperCase();
  if (!allowed.has(normalized)) throw new RangeError(`gate intent is not supported: ${value}`);
  return normalized as Intent;
}

function parseGateResult(value: unknown, allowed: ReadonlySet<string>): GateResult {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('gate output must be an object');
  }
  const candidate = value as Record<string, unknown>;
  if (typeof candidate.shouldRespond !== 'boolean') {
    throw new TypeError('gate shouldRespond must be a boolean');
  }
  if (typeof candidate.shouldRetrieve !== 'boolean') {
    throw new TypeError('gate shouldRetrieve must be a boolean');
  }
  return Object.freeze({
    intent: parseIntent(candidate.intent, allowed),
    shouldRespond: candidate.shouldRespond,
    shouldRetrieve: candidate.shouldRetrieve,
  });
}

function resolveIntents(customIntents: Iterable<string> | undefined): ReadonlySet<string> {
  const allowed = new Set(INTENT_SET);
  if (customIntents === undefined) return allowed;
  if (typeof customIntents === 'string' || customIntents === null
    || typeof customIntents[Symbol.iterator] !== 'function') {
    throw new TypeError('customIntents must be an iterable of strings');
  }
  for (const value of customIntents) {
    if (typeof value !== 'string') throw new TypeError('custom intents must be strings');
    const normalized = value.trim().toUpperCase();
    if (!CUSTOM_INTENT.test(normalized)) {
      throw new RangeError('custom intent must match [A-Z][A-Z0-9_]{0,63}');
    }
    allowed.add(normalized);
  }
  return allowed;
}

function resolveLimits(input: Partial<EngineLimits> | undefined): Readonly<EngineLimits> {
  const limits = { ...DEFAULT_ENGINE_LIMITS, ...(input ?? {}) };
  for (const [name, value] of Object.entries(limits)) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative safe integer`);
    }
  }
  if (limits.maxMessageChars === 0
    || limits.maxHistoryMessages === 0
    || limits.maxHistoryChars === 0
    || limits.maxPromptChars === 0
    || limits.maxOutputChars === 0
    || limits.maxQueuedTurns === 0) {
    throw new RangeError('message, history, prompt, output, and queued-turn limits must be positive');
  }
  return Object.freeze(limits);
}

function validateHistory(
  value: unknown,
  limits: Readonly<EngineLimits>,
): asserts value is ReadonlyArray<DeepReadonly<Message>> {
  if (!Array.isArray(value)) throw new TypeError('store history must be an array');
  if (value.length > limits.maxHistoryMessages) {
    throw new RangeError(`history exceeds ${limits.maxHistoryMessages} messages`);
  }
  let characters = 0;
  for (const [index, message] of value.entries()) {
    if (message === null || typeof message !== 'object' || Array.isArray(message)) {
      throw new TypeError(`history message ${index} must be an object`);
    }
    const candidate = message as Record<string, unknown>;
    if (candidate.role !== 'user' && candidate.role !== 'assistant') {
      throw new RangeError(`history message ${index} has an unsupported role`);
    }
    if (typeof candidate.content !== 'string') {
      throw new TypeError(`history message ${index} content must be a string`);
    }
    characters += candidate.content.length;
    if (characters > limits.maxHistoryChars) {
      throw new RangeError(`history exceeds ${limits.maxHistoryChars} characters`);
    }
  }
}

function validateControl(control: AdapterControl): void {
  const signal = control.signal;
  if (signal !== undefined
    && (signal === null
      || typeof signal !== 'object'
      || typeof signal.aborted !== 'boolean'
      || typeof signal.addEventListener !== 'function'
      || typeof signal.removeEventListener !== 'function')) {
    throw new TypeError('signal must be AbortSignal-compatible');
  }
  if (control.deadline !== undefined
    && (!(control.deadline instanceof Date) || !Number.isFinite(control.deadline.getTime()))) {
    throw new TypeError('deadline must be a valid Date');
  }
}

function abortError(signal: AbortSignalLike): Error {
  const error = new Error('turn aborted', { cause: signal.reason });
  error.name = 'AbortError';
  return error;
}

function timeoutError(): Error {
  const error = new Error('turn deadline exceeded');
  error.name = 'TimeoutError';
  return error;
}

function stopIfNeeded(control: AdapterControl): void {
  if (control.signal?.aborted) throw abortError(control.signal);
  if (control.deadline !== undefined && Date.now() >= control.deadline.getTime()) {
    throw timeoutError();
  }
}

function isControlError(value: unknown): boolean {
  if (value === null || typeof value !== 'object') return false;
  const name = (value as { readonly name?: unknown }).name;
  return name === 'AbortError' || name === 'TimeoutError';
}

function snapshotCallbacks(callbacks: TurnCallbacks): Readonly<TurnCallbacks> {
  if (callbacks === null || typeof callbacks !== 'object') {
    throw new TypeError('callbacks must be an object');
  }
  const signal = callbacks.signal;
  const deadline = callbacks.deadline;
  const onToken = callbacks.onToken;
  const rawControl: AdapterControl = {
    ...(signal === undefined ? {} : { signal }),
    ...(deadline === undefined ? {} : { deadline }),
  };
  validateControl(rawControl);
  if (onToken !== undefined && typeof onToken !== 'function') {
    throw new TypeError('onToken must be a function');
  }
  const snapshot: TurnCallbacks = {
    ...rawControl,
    ...(deadline === undefined ? {} : { deadline: new Date(deadline.getTime()) }),
    ...(onToken === undefined ? {} : { onToken }),
  };
  return Object.freeze(snapshot);
}

const MAX_TIMER_DELAY_MS = 2_147_483_647;
const RUNTIME_TIMERS = globalThis as unknown as {
  setTimeout(handler: () => void, delay: number): unknown;
  clearTimeout(handle: unknown): void;
};

function waitWithControl<Result>(
  work: PromiseLike<Result>,
  control: AdapterControl,
): Promise<Result> {
  try {
    stopIfNeeded(control);
  } catch (error) {
    return Promise.reject(error);
  }

  return new Promise<Result>((resolve, reject) => {
    let finished = false;
    let timer: unknown;
    let listening = false;
    const signal = control.signal;

    const cleanup = (): void => {
      if (timer !== undefined) RUNTIME_TIMERS.clearTimeout(timer);
      if (listening && signal !== undefined) {
        try {
          signal.removeEventListener('abort', onAbort);
        } catch {
          // A broken signal must not mask the operation's settled result.
        }
      }
    };
    const settle = (operation: () => void): void => {
      if (finished) return;
      finished = true;
      cleanup();
      operation();
    };
    const resolveOnce = (value: Result): void => { settle(() => resolve(value)); };
    const rejectOnce = (error: unknown): void => { settle(() => reject(error)); };
    const onAbort = (): void => {
      if (signal !== undefined) rejectOnce(abortError(signal));
    };
    const scheduleDeadline = (): void => {
      const deadline = control.deadline;
      if (deadline === undefined || finished) return;
      const remaining = deadline.getTime() - Date.now();
      if (remaining <= 0) {
        rejectOnce(timeoutError());
        return;
      }
      timer = RUNTIME_TIMERS.setTimeout(
        scheduleDeadline,
        Math.min(remaining, MAX_TIMER_DELAY_MS),
      );
    };

    Promise.resolve(work).then(resolveOnce, rejectOnce);
    try {
      if (signal !== undefined) {
        listening = true;
        signal.addEventListener('abort', onAbort, { once: true });
        if (signal.aborted) onAbort();
      }
      scheduleDeadline();
    } catch (error) {
      rejectOnce(error);
    }
  });
}

function closeIterator(iterator: AsyncIterator<unknown>): void {
  try {
    const completion = iterator.return?.();
    if (completion !== undefined) void Promise.resolve(completion).catch(() => undefined);
  } catch {
    // Iterator cleanup is best-effort and must not mask the turn failure.
  }
}

function adapterControl(control: AdapterControl): AdapterControl {
  return {
    ...(control.signal === undefined ? {} : { signal: control.signal }),
    ...(control.deadline === undefined ? {} : { deadline: new Date(control.deadline.getTime()) }),
  };
}

function adapterInput<Input extends object>(input: Input, control: AdapterControl): Readonly<Input & AdapterControl> {
  const snapshot = readonlySnapshot(input);
  return Object.freeze({ ...snapshot, ...adapterControl(control) }) as Readonly<Input & AdapterControl>;
}

function parseCandidates(value: unknown, limits: Readonly<EngineLimits>): MemoryCandidate[] {
  if (!Array.isArray(value)) throw new TypeError('memory retriever must return an array');
  if (value.length > limits.maxCandidates) {
    throw new RangeError(`memory retriever returned more than ${limits.maxCandidates} candidates`);
  }
  let textChars = 0;
  for (const [index, candidate] of value.entries()) {
    if (candidate === null || typeof candidate !== 'object' || Array.isArray(candidate)) {
      throw new TypeError(`memory candidate ${index} must be an object`);
    }
    const text = (candidate as Record<string, unknown>).text;
    if (typeof text !== 'string') throw new TypeError(`memory candidate ${index} text must be a string`);
    textChars += text.length;
    if (textChars > limits.maxMemoryChars) {
      throw new RangeError(`memory evidence exceeds ${limits.maxMemoryChars} characters`);
    }
  }
  return value as MemoryCandidate[];
}

function parseAppraisalResult(value: unknown): AppraiseTurnResult {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError('appraisal policy must return an AppraiseTurnResult object');
  }
  const candidate = value as Record<string, unknown>;
  if (candidate.mood === undefined
    || candidate.baselineValence === undefined
    || candidate.occCarry === undefined) {
    throw new TypeError('appraisal policy result is missing state fields');
  }
  const validated = createAffectState({
    mood: candidate.mood as NonNullable<AffectState['mood']>,
    baselineValence: candidate.baselineValence as number,
    occCarry: candidate.occCarry as Readonly<Record<string, number>>,
  });
  const emotions = candidate.activeEmotions;
  if (emotions === null || typeof emotions !== 'object' || Array.isArray(emotions)) {
    throw new TypeError('appraisal policy activeEmotions must be an object');
  }
  const activeEmotions: Record<string, number> = {};
  for (const [name, intensity] of Object.entries(emotions)) {
    if (!name) throw new RangeError('appraisal emotion names must not be empty');
    if (typeof intensity !== 'number' || !Number.isFinite(intensity)
      || intensity < 0 || intensity > 1) {
      throw new RangeError(`appraisal emotion ${name} must be finite and between 0 and 1`);
    }
    activeEmotions[name] = intensity;
  }
  return {
    mood: validated.mood,
    activeEmotions,
    occCarry: { ...validated.occCarry },
    baselineValence: validated.baselineValence,
  };
}

/** Injected gate → retrieval → appraisal → prompt → generation pipeline. */
export class AffectEngine {
  private readonly model: ModelAdapter;
  private readonly store: StateStore;
  private readonly retriever: MemoryRetriever | undefined;
  private readonly instruction: string;
  private readonly promptPolicy: PromptPolicy | undefined;
  private readonly turnShapePolicy: TurnShapePolicy | undefined;
  private readonly presenceLabels: PresenceLabels;
  private readonly stateFactory: (() => AffectState) | undefined;
  private readonly retrievalLimit: number;
  private readonly now: (() => Date) | undefined;
  private readonly limits: Readonly<EngineLimits>;
  private readonly gateFallback: GateResult | undefined;
  private readonly allowedIntents: ReadonlySet<string>;
  private readonly appraisalPolicy: AppraisalPolicy;
  private tail: Promise<void> = Promise.resolve();
  private admittedTurns = 0;

  constructor(options: AffectEngineOptions) {
    if (options === null || typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const {
      model,
      store,
      retriever,
      instruction = '',
      promptPolicy,
      turnShapePolicy,
      presenceLabels,
      stateFactory,
      retrievalLimit: limit = 6,
      gateFallback,
      customIntents,
      appraisalPolicy,
      limits,
      now,
    } = options;
    if (!Number.isSafeInteger(limit) || limit < 0) {
      throw new RangeError('retrievalLimit must be a non-negative safe integer');
    }
    if (typeof instruction !== 'string') {
      throw new TypeError('instruction must be a string');
    }
    this.limits = resolveLimits(limits);
    if (this.limits.maxCandidates > 0 && limit > this.limits.maxCandidates) {
      throw new RangeError('retrievalLimit must not exceed limits.maxCandidates');
    }
    this.allowedIntents = resolveIntents(customIntents);
    if (appraisalPolicy !== undefined && typeof appraisalPolicy !== 'function') {
      throw new TypeError('appraisalPolicy must be a function');
    }
    if (stateFactory !== undefined && typeof stateFactory !== 'function') {
      throw new TypeError('stateFactory must be a function');
    }
    if (now !== undefined && typeof now !== 'function') {
      throw new TypeError('now must be a function');
    }
    this.model = model;
    this.store = store;
    this.retriever = retriever;
    this.instruction = instruction;
    this.promptPolicy = promptPolicy === undefined
      ? undefined
      : readonlySnapshot(promptPolicy) as PromptPolicy;
    this.turnShapePolicy = turnShapePolicy === undefined
      ? undefined
      : readonlySnapshot(turnShapePolicy) as TurnShapePolicy;
    this.presenceLabels = presenceLabels === undefined
      ? DEFAULT_PRESENCE_LABELS
      : readonlySnapshot(presenceLabels) as PresenceLabels;
    this.stateFactory = stateFactory;
    this.retrievalLimit = limit;
    this.now = now;
    this.appraisalPolicy = appraisalPolicy ?? DEFAULT_APPRAISAL_POLICY;
    this.gateFallback = gateFallback === undefined
      ? undefined
      : parseGateResult(gateFallback, this.allowedIntents);
  }

  turn(message: string, callbacks: TurnCallbacks = {}): Promise<TurnResult> {
    if (typeof message !== 'string') return Promise.reject(new TypeError('message must be a string'));
    if (!stripPyWhitespace(message)) {
      return Promise.reject(new RangeError('message must not be empty'));
    }
    if (message.length > this.limits.maxMessageChars) {
      return Promise.reject(new RangeError(`message exceeds ${this.limits.maxMessageChars} characters`));
    }
    let control: Readonly<TurnCallbacks>;
    try {
      control = snapshotCallbacks(callbacks);
      stopIfNeeded(control);
    } catch (error) {
      return Promise.reject(error);
    }
    if (this.admittedTurns >= this.limits.maxQueuedTurns) {
      return Promise.reject(new RangeError('maximum queued turns exceeded'));
    }

    this.admittedTurns += 1;
    const task = this.tail.then(() => this.store.transaction(
      (transaction) => this.runTurn(transaction, message, control),
    ));
    this.tail = task.then(() => undefined, () => undefined);
    return task.finally(() => { this.admittedTurns -= 1; });
  }

  async presence(cognition: CognitionState = {}): Promise<PresenceVector> {
    const stored = await this.store.loadState();
    const state = createAffectState(stored ?? this.stateFactory?.() ?? {});
    return readonlySnapshot(
      buildPresenceVector(state, cognition, this.presenceLabels),
    ) as PresenceVector;
  }

  private currentTime(): Date {
    const now = this.now?.() ?? new Date();
    if (!(now instanceof Date) || !Number.isFinite(now.getTime())) {
      throw new TypeError('now must return a valid Date');
    }
    return new Date(now.getTime());
  }

  private async runTurn(
    transaction: StateTransaction,
    message: string,
    callbacks: TurnCallbacks,
  ): Promise<TurnResult> {
    stopIfNeeded(callbacks);
    const history = readonlySnapshot(await transaction.listMessages());
    validateHistory(history, this.limits);
    const stored = await transaction.loadState();
    const state = readonlySnapshot(createAffectState(
      stored ?? this.stateFactory?.() ?? {},
    ));
    const adapterState = readonlySnapshot(createAffectState({ ...state, carriedThought: null }));
    stopIfNeeded(callbacks);

    let gate: GateResult;
    let gateFallback = false;
    try {
      const rawGate = await waitWithControl(this.model.gate(adapterInput({
        message,
        history,
        state: adapterState,
      }, callbacks)), callbacks);
      gate = parseGateResult(rawGate, this.allowedIntents);
    } catch (error) {
      if (isControlError(error)) throw error;
      stopIfNeeded(callbacks);
      if (this.gateFallback === undefined) throw error;
      gate = this.gateFallback;
      gateFallback = true;
    }

    await transaction.appendMessage({ role: 'user', content: message });
    if (!gate.shouldRespond) {
      return readonlySnapshot({
        status: 'silent' as const,
        text: null,
        intent: gate.intent,
        mood: state.mood,
        emotions: {},
        memories: [],
        decoding: null,
        gateFallback,
      }) as TurnResult;
    }

    const now = this.currentTime();
    let candidates: MemoryCandidate[] = [];
    const retrievalLimit = this.retrievalLimit;
    if (gate.shouldRetrieve
      && this.retriever
      && retrievalLimit > 0
      && this.limits.maxCandidates > 0
      && this.limits.maxMemories > 0
      && this.limits.maxMemoryChars > 0) {
      const rawCandidates = await waitWithControl(this.retriever.retrieve(adapterInput({
        query: message,
        history,
        state: adapterState,
        limit: retrievalLimit,
        now: new Date(now.getTime()),
      }, callbacks)), callbacks);
      candidates = parseCandidates(rawCandidates, this.limits);
    }
    const memories: RankedMemory[] = rankCandidates(candidates, {
      limit: Math.min(retrievalLimit, this.limits.maxMemories),
      now,
      moodValence: state.mood.valence,
    });

    const appraisal = readonlySnapshot(parseAppraisalResult(this.appraisalPolicy(readonlySnapshot({
      mood: state.mood,
      personality: state.personality,
      goals: state.goals,
      stageInt: stageInt(state.relationship.stage),
      baselineValence: state.baselineValence,
      attachmentLonging: state.attachment.longing,
      intent: gate.intent,
      occCarry: state.occCarry,
      expectation: state.expectation,
      message,
    }))));
    const evolved = readonlySnapshot(createAffectState({
      ...state,
      mood: appraisal.mood,
      baselineValence: appraisal.baselineValence,
      occCarry: appraisal.occCarry,
    }));
    await transaction.saveState(evolved);

    const decoding = decodingParams(evolved.mood);
    const turnDirective = buildTurnShapeDirective(
      {
        mood: evolved.mood,
        intent: gate.intent,
        history,
        emotions: appraisal.activeEmotions,
      },
      this.turnShapePolicy,
    );
    const promptContext = {
      instruction: this.instruction,
      state: { mood: evolved.mood },
      emotions: appraisal.activeEmotions,
      decoding,
      turnDirective,
    };
    const systemPrompt = this.promptPolicy
      ? composePrompt(promptContext, this.promptPolicy)
      : composePrompt(promptContext);
    if (systemPrompt.length > this.limits.maxPromptChars) {
      throw new RangeError(`system prompt exceeds ${this.limits.maxPromptChars} characters`);
    }
    const untrustedContext = buildUntrustedContext(evolved, memories, {
      maxChars: this.limits.maxMemoryChars,
      surfaceCarriedThought: history.length === 0,
    });
    const generationState = readonlySnapshot(createAffectState({
      ...evolved,
      carriedThought: null,
    }));

    const chunks: string[] = [];
    let accumulated = '';
    const generation = this.model.generate(adapterInput({
      message,
      systemPrompt,
      history,
      state: generationState,
      intent: gate.intent,
      emotions: appraisal.activeEmotions,
      decoding,
      untrustedContext,
    }, callbacks));
    const iterator = generation[Symbol.asyncIterator]();
    let iteratorDone = false;
    try {
      while (true) {
        const result = await waitWithControl(iterator.next(), callbacks);
        if (result.done) {
          iteratorDone = true;
          break;
        }
        const chunk = result.value;
        if (typeof chunk !== 'string') throw new TypeError('model adapters must yield strings');
        if (accumulated.length + chunk.length > this.limits.maxOutputChars) {
          throw new RangeError(`model output exceeds ${this.limits.maxOutputChars} characters`);
        }
        chunks.push(chunk);
        accumulated += chunk;
        if (callbacks.onToken !== undefined) {
          await waitWithControl(
            Promise.resolve(callbacks.onToken(chunk, accumulated)),
            callbacks,
          );
        }
      }
    } finally {
      if (!iteratorDone) closeIterator(iterator);
    }
    const text = chunks.join('');
    await transaction.appendMessage({ role: 'assistant', content: text });
    return readonlySnapshot({
      status: 'responded' as const,
      text,
      intent: gate.intent,
      mood: evolved.mood,
      emotions: appraisal.activeEmotions,
      memories,
      decoding,
      gateFallback,
    }) as TurnResult;
  }
}
