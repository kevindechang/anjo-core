/**
 * Seeded property and fuzz tests over the kernel's stated invariants.
 *
 * Mirrors python/tests/test_properties.py. Uses a small deterministic PRNG
 * rather than a property-testing dependency, so a failure is re-runnable by
 * anyone from the seed alone.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_APPRAISAL_GOALS,
  appraiseTurn,
  applyLengthFactor,
  decodingParams,
  rankCandidates,
  type AppraisalGoals,
  type MemoryCandidate,
  type PadMood,
  type Personality,
} from '../src/index.js';

const SEED = 20260820;
const CASES = 2_000;

/** mulberry32: 32-bit, deterministic, and short enough to audit here. */
function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const between = (rng: () => number, low: number, high: number): number =>
  low + rng() * (high - low);

function randomMood(rng: () => number): PadMood {
  return { valence: between(rng, -1, 1), arousal: between(rng, -1, 1), dominance: between(rng, -1, 1) };
}

function randomPersonality(rng: () => number): Personality {
  return { O: rng(), C: rng(), E: rng(), A: rng(), N: rng() };
}

function randomGoals(rng: () => number): AppraisalGoals {
  return {
    rapport: rng(), intellectual: rng(), autonomy: rng(), respect: rng(), honesty: rng(),
  };
}

const INTENTS = ['ABUSE', 'APOLOGY', 'VULNERABILITY', 'CURIOSITY', 'CHALLENGE', 'NEGLECT', 'CASUAL'];

function assertInRange(mood: PadMood, baselineValence: number, carry: Record<string, number>): void {
  for (const axis of [mood.valence, mood.arousal, mood.dominance]) {
    assert.ok(axis >= -1 && axis <= 1, `PAD axis escaped [-1, 1]: ${axis}`);
  }
  assert.ok(baselineValence >= -1 && baselineValence <= 1);
  for (const value of Object.values(carry)) assert.ok(value >= 0 && value <= 1);
}

test('one appraised turn never leaves the declared domains', () => {
  const rng = makeRng(SEED);
  for (let i = 0; i < CASES; i += 1) {
    const result = appraiseTurn({
      mood: randomMood(rng),
      personality: randomPersonality(rng),
      goals: randomGoals(rng),
      stageInt: 1 + Math.floor(rng() * 5),
      baselineValence: between(rng, -1, 1),
      attachmentLonging: rng(),
      intent: INTENTS[Math.floor(rng() * INTENTS.length)] as string,
    });
    assertInRange(result.mood, result.baselineValence, result.occCarry);
    for (const value of Object.values(result.activeEmotions)) {
      assert.ok(value >= 0 && value <= 1);
    }
  }
});

test('a 200-turn adversarial walk never diverges', () => {
  const rng = makeRng(SEED);
  const worstCase = ['ABUSE', 'CURIOSITY', 'VULNERABILITY', 'ABUSE'];
  for (let run = 0; run < 40; run += 1) {
    let mood = randomMood(rng);
    let baselineValence = between(rng, -1, 1);
    let carry: Record<string, number> = {};
    const personality = randomPersonality(rng);
    const goals = randomGoals(rng);
    for (let turn = 0; turn < 200; turn += 1) {
      const result = appraiseTurn({
        mood,
        personality,
        goals,
        stageInt: 3,
        baselineValence,
        attachmentLonging: 0,
        intent: worstCase[Math.floor(rng() * worstCase.length)] as string,
        occCarry: carry,
      });
      ({ mood, baselineValence } = result);
      carry = result.occCarry;
      assertInRange(mood, baselineValence, carry);
    }
  }
});

test('appraisal is deterministic for identical input', () => {
  const rng = makeRng(SEED);
  for (let i = 0; i < 200; i += 1) {
    const input = {
      mood: randomMood(rng),
      personality: randomPersonality(rng),
      goals: DEFAULT_APPRAISAL_GOALS,
      stageInt: 2,
      baselineValence: between(rng, -1, 1),
      attachmentLonging: rng(),
      intent: INTENTS[Math.floor(rng() * INTENTS.length)] as string,
    };
    assert.deepEqual(appraiseTurn(input), appraiseTurn(input));
  }
});

test('the length factor can only shorten a budget', () => {
  const rng = makeRng(SEED);
  for (let i = 0; i < CASES; i += 1) {
    const budget = 1 + Math.floor(rng() * 100_000);
    const shortened = applyLengthFactor(budget, randomMood(rng));
    assert.ok(shortened > 0 && shortened <= budget);
  }
});

test('decoding stays inside the published envelope', () => {
  const rng = makeRng(SEED);
  for (let i = 0; i < CASES; i += 1) {
    const params = decodingParams(randomMood(rng));
    assert.ok(params.temperature >= 0.72 && params.temperature <= 1.18);
    assert.equal(params.topP, 0.97);
  }
});

test('ranking dedupes, sorts, respects the limit, and ignores input order', () => {
  const rng = makeRng(SEED);
  for (let i = 0; i < 300; i += 1) {
    const count = 1 + Math.floor(rng() * 24);
    const candidates: MemoryCandidate[] = [];
    for (let m = 0; m < count; m += 1) {
      candidates.push({
        id: `m${Math.floor(rng() * count)}`,
        text: 't',
        distance: between(rng, 0, 2),
        daysAgo: between(rng, 0, 400),
        episode: rng() < 0.3,
        significance: rng(),
        recallCount: Math.floor(rng() * 1000),
        emotionalValence: between(rng, -1, 1),
      });
    }
    const limit = Math.floor(rng() * 10);
    const ranked = rankCandidates(candidates, { limit });
    const ids = ranked.map((item) => item.id);
    assert.equal(new Set(ids).size, ids.length, 'duplicate ids survived ranking');
    assert.ok(ids.length <= limit);
    const scores = ranked.map((item) => item.score);
    assert.deepEqual(scores, [...scores].sort((a, b) => b - a));

    const unique = new Map(candidates.map((c) => [c.id, c]));
    const expected = rankCandidates(unique.values(), { limit: 5 }).map((m) => m.id);
    const shuffled = [...unique.values()].sort(() => rng() - 0.5);
    assert.deepEqual(rankCandidates(shuffled, { limit: 5 }).map((m) => m.id), expected);
  }
});
