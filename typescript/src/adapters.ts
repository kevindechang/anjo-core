import type {
  CompanionState,
  DeepReadonly,
  GateInput,
  GateResult,
  GenerateInput,
  MemoryCandidate,
  MemoryRetriever,
  Message,
  ModelAdapter,
  RetrievalInput,
  StateStore,
  StateTransaction,
} from './contracts.js';
import { cloneValue, readonlySnapshot } from './internal/snapshot.js';

export interface InMemoryStoreOptions {
  state?: CompanionState;
  messages?: ReadonlyArray<Message>;
}

/**
 * Process-local state and transcript store with defensive copies.
 * Direct reads expose the last committed snapshot while a transaction stages private changes.
 */
export class InMemoryStore implements StateStore {
  private state: CompanionState | null;
  private readonly messages: Message[];
  private transactionTail: Promise<void> = Promise.resolve();

  constructor(options: InMemoryStoreOptions = {}) {
    this.state = options.state === undefined ? null : cloneValue(options.state);
    this.messages = [...cloneValue(options.messages ?? [])];
  }

  private enqueue<Result>(operation: () => Promise<Result>): Promise<Result> {
    const task = this.transactionTail.then(operation);
    this.transactionTail = task.then(() => undefined, () => undefined);
    return task;
  }

  loadState(): Promise<DeepReadonly<CompanionState> | null> {
    return Promise.resolve(this.state === null ? null : readonlySnapshot(this.state));
  }

  saveState(state: DeepReadonly<CompanionState>): Promise<void> {
    const snapshot = cloneValue(state);
    return this.enqueue(async () => { this.state = snapshot; });
  }

  listMessages(): Promise<ReadonlyArray<DeepReadonly<Message>>> {
    return Promise.resolve(readonlySnapshot(this.messages));
  }

  appendMessage(message: DeepReadonly<Message>): Promise<void> {
    const snapshot = cloneValue(message);
    return this.enqueue(async () => { this.messages.push(snapshot); });
  }

  async transaction<Result>(
    operation: (transaction: StateTransaction) => Promise<Result>,
  ): Promise<Result> {
    return this.enqueue(async () => {
      let workingState = this.state === null ? null : cloneValue(this.state);
      const workingMessages = cloneValue(this.messages);
      const transaction: StateTransaction = {
        loadState: async () => workingState === null ? null : readonlySnapshot(workingState),
        saveState: async (state) => { workingState = cloneValue(state); },
        listMessages: async () => readonlySnapshot(workingMessages),
        appendMessage: async (message) => { workingMessages.push(cloneValue(message)); },
      };
      const result = await operation(transaction);
      this.state = workingState === null ? null : cloneValue(workingState);
      this.messages.splice(0, this.messages.length, ...cloneValue(workingMessages));
      return result;
    });
  }
}

/** Fixed candidate source for tests, examples, and offline prototyping. */
export class InMemoryRetriever implements MemoryRetriever {
  private readonly candidates: MemoryCandidate[];
  private readonly requestLog: RetrievalInput[] = [];

  get requests(): ReadonlyArray<DeepReadonly<RetrievalInput>> {
    return readonlySnapshot(this.requestLog);
  }

  constructor(candidates: ReadonlyArray<MemoryCandidate> = []) {
    this.candidates = [...cloneValue(candidates)];
  }

  async retrieve(input: RetrievalInput): Promise<ReadonlyArray<MemoryCandidate>> {
    this.requestLog.push({
      ...input,
      history: cloneValue(input.history),
      state: cloneValue(input.state),
      now: new Date(input.now.getTime()),
    });
    return readonlySnapshot(this.candidates);
  }
}

export interface ScriptedModelOptions {
  gates?: ReadonlyArray<GateResult | Error>;
  responses?: ReadonlyArray<string | ReadonlyArray<string>>;
  defaultGate?: GateResult;
}

/** Credential-free model seam that consumes deterministic queued outputs. */
export class ScriptedModelAdapter implements ModelAdapter {
  private readonly gates: Array<GateResult | Error>;
  private readonly responses: string[][];
  private readonly defaultGate: GateResult;
  private readonly gateRequestLog: GateInput[] = [];
  private readonly generateRequestLog: GenerateInput[] = [];

  get gateRequests(): ReadonlyArray<DeepReadonly<GateInput>> {
    return readonlySnapshot(this.gateRequestLog);
  }

  get generateRequests(): ReadonlyArray<DeepReadonly<GenerateInput>> {
    return readonlySnapshot(this.generateRequestLog);
  }

  constructor(options: ScriptedModelOptions = {}) {
    this.gates = [...(options.gates ?? [])];
    this.responses = (options.responses ?? []).map((response) => (
      typeof response === 'string' ? [response] : [...response]
    ));
    this.defaultGate = options.defaultGate ?? {
      intent: 'CASUAL', shouldRespond: true, shouldRetrieve: false,
    };
  }

  async gate(input: GateInput): Promise<GateResult> {
    this.gateRequestLog.push(cloneValue(input));
    const next = this.gates.shift();
    if (next instanceof Error) throw next;
    return readonlySnapshot(next ?? this.defaultGate);
  }

  async *generate(input: GenerateInput): AsyncIterable<string> {
    this.generateRequestLog.push(cloneValue(input));
    const response = this.responses.shift();
    if (!response) throw new Error('no scripted response remains');
    for (const chunk of response) {
      await Promise.resolve();
      yield chunk;
    }
  }
}
