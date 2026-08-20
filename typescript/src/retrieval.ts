/**
 * Pure memory scoring and ranking, independent of storage or embeddings.
 *
 * The relevance x recency x salience decomposition follows Generative Agents
 * (Park et al. 2023), which sums those factors where this module multiplies
 * them, and decays recency exponentially where this module is linear to a
 * floor. Constant provenance is recorded in docs/foundations.md sections 6-7.
 */
import type { MemoryCandidate, RankedMemory } from './contracts.js';

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

/**
 * The numeric parameters of the retrieval scorer.
 *
 * Defaults reproduce the pinned cross-runtime contract. These are magnitudes
 * only: the *shape* of the curves -- linear recency, multiplicative
 * composition, log-compressed rehearsal -- is fixed here, and
 * docs/foundations.md section 6 records why each shape was chosen and what the
 * closest published comparable does instead.
 */
export interface RetrievalWeights {
  readonly recencyHorizonDays: number;
  readonly recencyFloor: number;
  readonly recencyFallback: number;
  readonly significanceWeight: number;
  readonly rehearsalWeight: number;
  readonly rehearsalCap: number;
  readonly episodeBonus: number;
  readonly congruenceThreshold: number;
  readonly congruenceNegativeMood: number;
  readonly congruencePositiveMood: number;
}

export const DEFAULT_RETRIEVAL_WEIGHTS: RetrievalWeights = Object.freeze({
  recencyHorizonDays: 60,
  recencyFloor: 0.4,
  recencyFallback: 0.7,
  significanceWeight: 0.03,
  rehearsalWeight: 0.006,
  rehearsalCap: 0.025,
  episodeBonus: 0.05,
  congruenceThreshold: 0.2,
  congruenceNegativeMood: 1.06,
  congruencePositiveMood: 1.03,
});

export function recencyWeight(
  daysAgo: number,
  weights: RetrievalWeights = DEFAULT_RETRIEVAL_WEIGHTS,
): number {
  if (!Number.isFinite(daysAgo)) throw new TypeError('daysAgo must be finite');
  return clamp(1 - daysAgo / weights.recencyHorizonDays, weights.recencyFloor, 1);
}

export function recencyWeightFromTimestamp(
  timestamp: string,
  now = new Date(),
  weights: RetrievalWeights = DEFAULT_RETRIEVAL_WEIGHTS,
): number {
  const time = Date.parse(timestamp);
  if (!Number.isFinite(time)) return weights.recencyFallback;
  return recencyWeight((now.getTime() - time) / 86_400_000, weights);
}

/**
 * Mood-congruent recall is Bower (1981); the threshold, the magnitudes, and the
 * negative/positive asymmetry are production-tuned and unsupported by any
 * citation in docs/foundations.md section 7.
 */
export function moodCongruenceFactor(
  memoryValence: number,
  moodValence: number,
  congruenceOn: boolean,
  weights: RetrievalWeights = DEFAULT_RETRIEVAL_WEIGHTS,
): number {
  if (!congruenceOn || memoryValence === 0) return 1;
  if ((memoryValence > 0) === (moodValence > 0)) {
    return moodValence < 0 ? weights.congruenceNegativeMood : weights.congruencePositiveMood;
  }
  return 1;
}

export function similarityFromDistance(distance: number): number {
  if (!Number.isFinite(distance)) throw new TypeError('distance must be finite');
  return clamp(1 - distance / 2, 0, 1);
}

export interface CandidateScoreInput {
  similarity: number;
  recency: number;
  episode: boolean;
  significance: number;
  recallCount: number;
  weights?: RetrievalWeights;
}

export function candidateScore(input: CandidateScoreInput): number {
  if (!Number.isFinite(input.similarity)) throw new TypeError('similarity must be finite');
  if (!Number.isFinite(input.recency)) throw new TypeError('recency must be finite');
  if (!Number.isFinite(input.significance)) throw new TypeError('significance must be finite');
  if (!Number.isSafeInteger(input.recallCount) || input.recallCount < 0) {
    throw new RangeError('recallCount must be a non-negative safe integer');
  }
  const weights = input.weights ?? DEFAULT_RETRIEVAL_WEIGHTS;
  const significance = clamp(input.significance, 0, 1);
  const recallCount = input.recallCount;
  const salience = 1 + significance * weights.significanceWeight
    + Math.min(weights.rehearsalCap, Math.log1p(recallCount) * weights.rehearsalWeight);
  return clamp(input.similarity, 0, 1) * clamp(input.recency, weights.recencyFloor, 1) * salience
    + (input.episode ? weights.episodeBonus : 0);
}

export interface RankCandidatesOptions {
  limit?: number;
  now?: Date;
  moodValence?: number;
  moodCongruence?: boolean;
  weights?: RetrievalWeights;
}

const ZONED_TIMESTAMP = /T.*(?:Z|[+-]\d{2}:\d{2})$/iu;

function validateCandidate(candidate: MemoryCandidate): void {
  if (typeof candidate.id !== 'string' || candidate.id.length === 0) {
    throw new RangeError('memory candidate id must not be empty');
  }
  if (typeof candidate.text !== 'string') throw new TypeError('memory candidate text must be a string');
  if (!Number.isFinite(candidate.distance) || candidate.distance < 0 || candidate.distance > 2) {
    throw new RangeError('memory candidate distance must be finite and between 0 and 2');
  }
  if (candidate.daysAgo !== undefined
    && (!Number.isFinite(candidate.daysAgo) || candidate.daysAgo < 0)) {
    throw new RangeError('memory candidate daysAgo must be finite and non-negative');
  }
  if (candidate.timestamp !== undefined && candidate.timestamp !== null) {
    if (typeof candidate.timestamp !== 'string') {
      throw new TypeError('memory candidate timestamp must be a string or null');
    }
    if (!ZONED_TIMESTAMP.test(candidate.timestamp) || !Number.isFinite(Date.parse(candidate.timestamp))) {
      throw new RangeError('memory candidate timestamp must be a valid zoned ISO timestamp');
    }
  }
  if (candidate.episode !== undefined && typeof candidate.episode !== 'boolean') {
    throw new TypeError('memory candidate episode must be a boolean');
  }
  if (candidate.significance !== undefined
    && (!Number.isFinite(candidate.significance)
      || candidate.significance < 0
      || candidate.significance > 1)) {
    throw new RangeError('memory candidate significance must be between 0 and 1');
  }
  if (candidate.recallCount !== undefined
    && (!Number.isSafeInteger(candidate.recallCount) || candidate.recallCount < 0)) {
    throw new RangeError('memory candidate recallCount must be a non-negative safe integer');
  }
  if (candidate.emotionalValence !== undefined
    && (!Number.isFinite(candidate.emotionalValence)
      || candidate.emotionalValence < -1
      || candidate.emotionalValence > 1)) {
    throw new RangeError('memory candidate emotionalValence must be between -1 and 1');
  }
}

export function scoreCandidate(
  candidate: MemoryCandidate,
  options: Omit<RankCandidatesOptions, 'limit'> = {},
): number {
  validateCandidate(candidate);
  const now = options.now ?? new Date();
  const weights = options.weights ?? DEFAULT_RETRIEVAL_WEIGHTS;
  const recency = candidate.daysAgo === undefined
    ? recencyWeightFromTimestamp(candidate.timestamp ?? '', now, weights)
    : recencyWeight(candidate.daysAgo, weights);
  const score = candidateScore({
    similarity: similarityFromDistance(candidate.distance ?? 0),
    recency,
    episode: candidate.episode ?? false,
    significance: candidate.significance ?? 0.5,
    recallCount: candidate.recallCount ?? 0,
    weights,
  });
  const moodValence = options.moodValence ?? 0;
  const congruenceOn =
    (options.moodCongruence ?? true) && Math.abs(moodValence) >= weights.congruenceThreshold;
  return score * moodCongruenceFactor(
    candidate.emotionalValence ?? 0,
    moodValence,
    congruenceOn,
    weights,
  );
}

export function rankCandidates(
  candidates: Iterable<MemoryCandidate>,
  options: RankCandidatesOptions = {},
): RankedMemory[] {
  const limit = options.limit ?? 4;
  if (!Number.isInteger(limit) || limit < 0) {
    throw new RangeError('limit must be a non-negative integer');
  }
  const best = new Map<string, RankedMemory>();
  for (const candidate of candidates) {
    const ranked = { ...candidate, score: scoreCandidate(candidate, options) };
    const previous = best.get(candidate.id);
    if (!previous || ranked.score > previous.score) best.set(candidate.id, ranked);
  }
  return [...best.values()]
    .sort((left, right) => right.score - left.score || compareCodePoints(left.id, right.id))
    .slice(0, limit);
}

function compareCodePoints(left: string, right: string): number {
  const leftIterator = left[Symbol.iterator]();
  const rightIterator = right[Symbol.iterator]();
  while (true) {
    const leftNext = leftIterator.next();
    const rightNext = rightIterator.next();
    if (leftNext.done || rightNext.done) {
      if (leftNext.done === rightNext.done) return 0;
      return leftNext.done ? -1 : 1;
    }
    const leftPoint = leftNext.value.codePointAt(0) as number;
    const rightPoint = rightNext.value.codePointAt(0) as number;
    if (leftPoint !== rightPoint) return leftPoint < rightPoint ? -1 : 1;
  }
}
