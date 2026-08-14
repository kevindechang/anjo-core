import type { MemoryCandidate, RankedMemory } from './contracts.js';

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function recencyWeight(daysAgo: number): number {
  if (!Number.isFinite(daysAgo)) throw new TypeError('daysAgo must be finite');
  return clamp(1 - daysAgo / 60, 0.4, 1);
}

export function recencyWeightFromTimestamp(timestamp: string, now = new Date()): number {
  const time = Date.parse(timestamp);
  if (!Number.isFinite(time)) return 0.7;
  return recencyWeight((now.getTime() - time) / 86_400_000);
}

export function moodCongruenceFactor(
  memoryValence: number,
  moodValence: number,
  congruenceOn: boolean,
): number {
  if (!congruenceOn || memoryValence === 0) return 1;
  if ((memoryValence > 0) === (moodValence > 0)) return moodValence < 0 ? 1.06 : 1.03;
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
}

export function candidateScore(input: CandidateScoreInput): number {
  if (!Number.isFinite(input.similarity)) throw new TypeError('similarity must be finite');
  if (!Number.isFinite(input.recency)) throw new TypeError('recency must be finite');
  if (!Number.isFinite(input.significance)) throw new TypeError('significance must be finite');
  if (!Number.isSafeInteger(input.recallCount) || input.recallCount < 0) {
    throw new RangeError('recallCount must be a non-negative safe integer');
  }
  const significance = clamp(input.significance, 0, 1);
  const recallCount = input.recallCount;
  const salience = 1 + significance * 0.03
    + Math.min(0.025, Math.log1p(recallCount) * 0.006);
  return clamp(input.similarity, 0, 1) * clamp(input.recency, 0.4, 1) * salience
    + (input.episode ? 0.05 : 0);
}

export interface RankCandidatesOptions {
  limit?: number;
  now?: Date;
  moodValence?: number;
  moodCongruence?: boolean;
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
  const recency = candidate.daysAgo === undefined
    ? recencyWeightFromTimestamp(candidate.timestamp ?? '', now)
    : recencyWeight(candidate.daysAgo);
  const score = candidateScore({
    similarity: similarityFromDistance(candidate.distance ?? 0),
    recency,
    episode: candidate.episode ?? false,
    significance: candidate.significance ?? 0.5,
    recallCount: candidate.recallCount ?? 0,
  });
  const moodValence = options.moodValence ?? 0;
  const congruenceOn = (options.moodCongruence ?? true) && Math.abs(moodValence) >= 0.2;
  return score * moodCongruenceFactor(
    candidate.emotionalValence ?? 0,
    moodValence,
    congruenceOn,
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
