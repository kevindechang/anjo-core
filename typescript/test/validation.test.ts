import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createCompanionState,
  applyLengthFactor,
  rankCandidates,
  recencyWeight,
  recencyWeightFromTimestamp,
  similarityFromDistance,
} from '../src/index.js';

test('length shaping validates and never increases the caller token budget', () => {
  const lowEnergy = { valence: 0, arousal: -0.8, dominance: 0 };
  assert.equal(applyLengthFactor(100, lowEnergy), 100);
  assert.equal(applyLengthFactor(1_000, lowEnergy), 600);
  assert.throws(() => applyLengthFactor(0, lowEnergy), /positive integer/i);
  assert.throws(() => applyLengthFactor(1.5, lowEnergy), /positive integer/i);
});

test('companion state validates finite values, ranges, counts, and OCC values', () => {
  assert.throws(() => createCompanionState({ mood: { valence: Number.NaN } }), /valence/i);
  assert.throws(() => createCompanionState({ personality: { O: 1.01 } }), /personality\.O/i);
  assert.throws(() => createCompanionState({ relationship: { trustScore: -0.1 } }), /trustScore/i);
  assert.throws(() => createCompanionState({ relationship: { sessionCount: 1.5 } }), /sessionCount/i);
  assert.throws(() => createCompanionState({ occCarry: { joy: Number.POSITIVE_INFINITY } }), /occCarry/i);
});

test('memory candidates validate finite ranges and require zoned timestamps', () => {
  assert.throws(() => rankCandidates([{ id: 'x', text: 'x', distance: Number.NaN }]), /distance/i);
  assert.throws(() => rankCandidates([{ id: 'x', text: 'x', distance: 0, significance: 2 }]), /significance/i);
  assert.throws(() => rankCandidates([{ id: 'x', text: 'x', distance: 0, recallCount: -1 }]), /recallCount/i);
  assert.throws(() => rankCandidates([{
    id: 'x', text: 'x', distance: 0, timestamp: '2026-08-14T12:00:00',
  }]), /timezone|zoned/i);
});

test('similarity and recency calculations clamp adversarial and future inputs', () => {
  assert.equal(similarityFromDistance(-5), 1);
  assert.equal(similarityFromDistance(99), 0);
  assert.equal(recencyWeight(-30), 1);
  assert.equal(recencyWeight(99_999), 0.4);
  assert.equal(
    recencyWeightFromTimestamp('2027-08-14T00:00:00Z', new Date('2026-08-14T00:00:00Z')),
    1,
  );
});

test('ranked memory ties use locale-independent Unicode code-point ordering', () => {
  const ranked = rankCandidates([
    { id: '\u{1F600}', text: 'emoji', distance: 0.5, daysAgo: 1 },
    { id: '\uE000', text: 'private use', distance: 0.5, daysAgo: 1 },
  ]);
  assert.deepEqual(ranked.map(({ id }) => id), ['\uE000', '\u{1F600}']);
});
