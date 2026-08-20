/**
 * The numeric seam: every coefficient is caller-owned data, not kernel behavior.
 *
 * Mirrors python/tests/test_parameters.py. The defaults must still reproduce
 * the cross-runtime contract, so exposing the knobs cannot move the baseline.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_AFFECT_DYNAMICS,
  DEFAULT_APPRAISAL_GOALS,
  DEFAULT_PERSONALITY,
  DEFAULT_RETRIEVAL_WEIGHTS,
  appraiseInput,
  appraiseTurn,
  decayMood,
  moodCongruenceFactor,
  moodInertia,
  occCarryDecay,
  rankCandidates,
  recencyWeight,
  recencyWeightFromTimestamp,
  scoreCandidate,
  type AffectDynamics,
  type MemoryCandidate,
  type RetrievalWeights,
} from '../src/index.js';

const NOW = new Date('2026-01-01T00:00:00Z');

function dynamics(overrides: Partial<AffectDynamics>): AffectDynamics {
  return { ...DEFAULT_AFFECT_DYNAMICS, ...overrides };
}

function weights(overrides: Partial<RetrievalWeights>): RetrievalWeights {
  return { ...DEFAULT_RETRIEVAL_WEIGHTS, ...overrides };
}

test('passing the defaults explicitly changes nothing', () => {
  const base = {
    mood: { valence: 0.4, arousal: 0.2, dominance: 0.1 },
    personality: DEFAULT_PERSONALITY,
    goals: DEFAULT_APPRAISAL_GOALS,
    stageInt: 3,
    baselineValence: 0.3,
    attachmentLonging: 0,
    intent: 'CURIOSITY',
  };
  assert.deepEqual(
    appraiseTurn(base),
    appraiseTurn({ ...base, dynamics: DEFAULT_AFFECT_DYNAMICS }),
  );
});

test('retrieval defaults are identical to omitting the weights', () => {
  const candidate: MemoryCandidate = {
    id: 'm', text: 't', distance: 0.6, daysAgo: 4,
    significance: 0.9, recallCount: 5, episode: true,
  };
  assert.equal(
    scoreCandidate(candidate, { now: NOW }),
    scoreCandidate(candidate, { now: NOW, weights: DEFAULT_RETRIEVAL_WEIGHTS }),
  );
});

test('inertia terms are caller-owned', () => {
  const flat = dynamics({ inertiaNeuroticism: 0, inertiaExtraversion: 0 });
  const anxious = { ...DEFAULT_PERSONALITY, N: 1, E: 0 };
  const calm = { ...DEFAULT_PERSONALITY, N: 0, E: 1 };
  assert.equal(moodInertia(anxious, flat), moodInertia(calm, flat));
  assert.ok(moodInertia(anxious) > moodInertia(calm));
});

test('zero inertia collapses mood onto the resting point', () => {
  const frozen = dynamics({ inertiaBase: 0, inertiaMin: 0, inertiaMax: 0 });
  assert.deepEqual(
    decayMood({ valence: 0.9, arousal: 0.9, dominance: 0.9 }, DEFAULT_PERSONALITY, 1, 0,
      undefined, frozen),
    { valence: 0, arousal: 0, dominance: 0 },
  );
});

test('the resting-dominance coefficient is caller-owned', () => {
  const raised = dynamics({ restingDominance: 1 });
  const plain = decayMood({ valence: 0, arousal: 0, dominance: 0 }, DEFAULT_PERSONALITY, 5, 0.5);
  const lifted = decayMood({ valence: 0, arousal: 0, dominance: 0 }, DEFAULT_PERSONALITY, 5, 0.5,
    undefined, raised);
  assert.ok(lifted.dominance > plain.dominance);
});

test('ambiguity amplification is caller-owned', () => {
  const off = dynamics({ ambiguityNegativeGain: 1, ambiguityPositiveGain: 1 });
  const mood = { valence: 0.3, arousal: 0, dominance: 0 };
  assert.equal(appraiseInput(mood, DEFAULT_APPRAISAL_GOALS, 'CASUAL', 0).mood.valence, 0.3328);
  assert.equal(appraiseInput(mood, DEFAULT_APPRAISAL_GOALS, 'CASUAL', 0, off).mood.valence, 0.32);
});

test('the baseline blend is caller-owned', () => {
  const instant = dynamics({ baselineRetention: 0, baselineIntake: 1 });
  const result = appraiseInput(
    { valence: 0, arousal: 0, dominance: 0 }, DEFAULT_APPRAISAL_GOALS, 'CURIOSITY', 0.9, instant,
  );
  assert.equal(result.baselineValence, result.mood.valence);
});

test('carry decay rates, fallback, and floor are caller-owned', () => {
  const tuning = dynamics({ carryDecay: { joy: 0.5 }, carryDecayDefault: 0.25 });
  const decayed = occCarryDecay({ joy: 1, admiration: 1 }, tuning);
  assert.equal(decayed.joy, 0.5);
  assert.equal(decayed.admiration, 0.25);

  assert.deepEqual(occCarryDecay({ joy: 0.06 }, dynamics({ carryFloor: 0.9 })), {});
  assert.ok(occCarryDecay({ joy: 0.06 }, dynamics({ carryFloor: 0 })).joy);
});

test('the carry floor reaches the appraiseTurn entry point', () => {
  const base = {
    mood: { valence: 0, arousal: 0, dominance: 0 },
    personality: DEFAULT_PERSONALITY,
    goals: DEFAULT_APPRAISAL_GOALS,
    stageInt: 1,
    baselineValence: 0,
    attachmentLonging: 0,
    // CASUAL would not do: its only signal is joy at exactly 0.05, which the
    // default floor excludes because the comparison is strictly greater-than.
    intent: 'CURIOSITY',
  };
  assert.deepEqual(appraiseTurn({ ...base, dynamics: dynamics({ carryFloor: 1 }) }).occCarry, {});
  assert.ok(Object.keys(appraiseTurn(base).occCarry).length > 0);
});

test('recency horizon, floor, and fallback are caller-owned', () => {
  assert.equal(recencyWeight(15), 0.75);
  assert.equal(recencyWeight(15, weights({ recencyHorizonDays: 10 })), 0.4);
  assert.equal(recencyWeight(1e9, weights({ recencyFloor: 0.1 })), 0.1);
  assert.equal(
    recencyWeightFromTimestamp('not-a-timestamp', NOW, weights({ recencyFallback: 0.2 })),
    0.2,
  );
});

test('the episode bonus is caller-owned', () => {
  const episodic: MemoryCandidate = { id: 'm', text: 't', distance: 0.5, daysAgo: 1, episode: true };
  const plain: MemoryCandidate = { id: 'm', text: 't', distance: 0.5, daysAgo: 1 };
  const none = weights({ episodeBonus: 0 });
  assert.ok(scoreCandidate(episodic, { now: NOW }) > scoreCandidate(plain, { now: NOW }));
  assert.equal(
    scoreCandidate(episodic, { now: NOW, weights: none }),
    scoreCandidate(plain, { now: NOW, weights: none }),
  );
});

test('the congruence threshold and magnitudes are caller-owned', () => {
  const candidate: MemoryCandidate = {
    id: 'm', text: 't', distance: 0.5, daysAgo: 1, emotionalValence: -0.5,
  };
  const eager = weights({ congruenceThreshold: 0, congruenceNegativeMood: 2 });
  const base = scoreCandidate(candidate, { now: NOW });
  assert.equal(scoreCandidate(candidate, { now: NOW, moodValence: -0.05 }), base);
  assert.equal(
    scoreCandidate(candidate, { now: NOW, moodValence: -0.05, weights: eager }),
    base * 2,
  );
  assert.equal(moodCongruenceFactor(0.5, -0.5, true, eager), 1);
});

test('weights reach the ranking entry point', () => {
  const candidates: MemoryCandidate[] = [
    { id: 'old', text: 't', distance: 0.4, daysAgo: 61 },
    { id: 'new', text: 't', distance: 0.5, daysAgo: 1 },
  ];
  const patient = weights({ recencyHorizonDays: 100_000 });
  assert.deepEqual(rankCandidates(candidates, { now: NOW }).map((m) => m.id), ['new', 'old']);
  assert.deepEqual(
    rankCandidates(candidates, { now: NOW, weights: patient }).map((m) => m.id),
    ['old', 'new'],
  );
});
