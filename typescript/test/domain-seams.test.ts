/**
 * The seams that let a non-companion domain reuse the kernel.
 *
 * Mirrors python/tests/test_domain_seams.py. Each test pins a promise the
 * README makes about generalization.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {
  DEFAULT_PRESENCE_LABELS,
  DEFAULT_STAGE_LADDER,
  UnknownStageError,
  baselineWeight,
  buildPresenceVector,
  buildTurnShapeDirective,
  createStageLadder,
  decayMood,
  expectationEmotions,
  ladderKnows,
  presenceLine,
  stageInt,
  type ExpectationCues,
  type PresenceLabels,
  type TurnShapePolicy,
} from '../src/index.js';
import { NEUTRAL_TURN_SHAPE_POLICY } from '../src/affect.js';
import { DEFAULT_PERSONALITY } from '../src/contracts.js';

const FACTION_LADDER = createStageLadder({
  stages: ['hostile', 'wary', 'neutral', 'friendly', 'sworn'],
  weights: [0, 0.15, 0.35, 0.6, 0.85],
});

test('a custom ladder replaces the conversational rungs', () => {
  assert.equal(stageInt('neutral', FACTION_LADDER), 3);
  assert.equal(baselineWeight(5, FACTION_LADDER), 0.85);
  assert.equal(ladderKnows(FACTION_LADDER, 'intimate'), false);
});

test('the default ladder floors unknown stages as pinned by the shared fixture', () => {
  assert.equal(stageInt('unknown'), 1);
  assert.equal(baselineWeight(stageInt('unknown')), 0);
  assert.equal(DEFAULT_STAGE_LADDER.strict, false);
});

test('a strict ladder surfaces a typo instead of silently flooring', () => {
  const strict = createStageLadder({
    stages: FACTION_LADDER.stages,
    weights: FACTION_LADDER.weights,
    strict: true,
  });
  assert.equal(stageInt('friendly', strict), 4);
  assert.throws(() => stageInt('freindly', strict), UnknownStageError);
});

test('out-of-range ordinals have no resting weight', () => {
  assert.equal(baselineWeight(0, FACTION_LADDER), 0);
  assert.equal(baselineWeight(99, FACTION_LADDER), 0);
  assert.throws(() => baselineWeight(1.5, FACTION_LADDER), TypeError);
});

test('malformed ladders are rejected', () => {
  assert.throws(() => createStageLadder({ stages: [], weights: [] }), RangeError);
  assert.throws(() => createStageLadder({ stages: ['a', 'b'], weights: [0.1] }), RangeError);
  assert.throws(() => createStageLadder({ stages: ['a', 'a'], weights: [0.1, 0.2] }), RangeError);
  assert.throws(() => createStageLadder({ stages: ['a'], weights: [1.5] }), RangeError);
  assert.throws(() => createStageLadder({ stages: ['a'], weights: [Number.NaN] }), RangeError);
});

test('the ladder changes the resting point of mood decay', () => {
  const mood = { valence: 0.5, arousal: 0, dominance: 0 };
  const onFaction = decayMood(mood, DEFAULT_PERSONALITY, 4, 0.4, FACTION_LADDER);
  const onDefault = decayMood(mood, DEFAULT_PERSONALITY, 4, 0.4);
  // Rung 4 weighs 0.60 on both ladders, so parity holds where the rungs agree.
  assert.equal(onFaction.valence, onDefault.valence);
  // Rung 5 differs (0.85 vs 0.70), so the domain ladder must diverge there.
  const highFaction = decayMood(mood, DEFAULT_PERSONALITY, 5, 0.4, FACTION_LADDER);
  const highDefault = decayMood(mood, DEFAULT_PERSONALITY, 5, 0.4);
  assert.notEqual(highFaction.valence, highDefault.valence);
});

test('a domain can replace the expectation vocabulary entirely', () => {
  const cues: ExpectationCues = {
    negativeExpected: ['failing'],
    positiveCurrent: ['green'],
    negativeCurrent: ['red'],
    expectedResolution: ['deploy'],
  };
  assert.deepEqual(expectationEmotions('the suite was failing', 'build is green', cues), {
    relief: 0.45,
    surprise: 0.35,
  });
  // The English preset finds nothing in the same strings.
  assert.deepEqual(expectationEmotions('the suite was failing', 'build is green'), {});
});

test('the reference policy still suppresses upbeat cues after vulnerability', () => {
  const upbeat = { valence: 0.6, arousal: 0.5, dominance: 0.4 };
  const suppressed = buildTurnShapeDirective(
    { mood: upbeat, intent: 'VULNERABILITY' },
    NEUTRAL_TURN_SHAPE_POLICY,
  );
  const allowed = buildTurnShapeDirective(
    { mood: upbeat, intent: 'CASUAL' },
    NEUTRAL_TURN_SHAPE_POLICY,
  );
  assert.ok(!suppressed.includes('momentum'));
  assert.ok(allowed.includes('momentum'));
});

test('a policy that declares no suppression suppresses nothing', () => {
  const upbeat = { valence: 0.6, arousal: 0.5, dominance: 0.4 };
  const policy: TurnShapePolicy = {
    heading: 'Bark shape',
    baseRules: ['One clause only.'],
    afterQuestionRule: 'End on a statement.',
    defaultCloseRule: 'End on a statement.',
    ambivalenceRule: 'Let the conflict show.',
    moodCues: { exuberant: 'Openly pleased.' },
  };
  const directive = buildTurnShapeDirective({ mood: upbeat, intent: 'VULNERABILITY' }, policy);
  assert.ok(directive.includes('Openly pleased.'));
});

test('a domain can suppress cues for its own intents', () => {
  const upbeat = { valence: 0.6, arousal: 0.5, dominance: 0.4 };
  const policy: TurnShapePolicy = {
    ...NEUTRAL_TURN_SHAPE_POLICY,
    suppressedMoodCues: { PLAYER_DIED: ['exuberant'] },
  };
  const directive = buildTurnShapeDirective({ mood: upbeat, intent: 'player_died' }, policy);
  assert.ok(!directive.includes('momentum'));
});

test('a domain renders its own presence wording', () => {
  const labels: PresenceLabels = { ...DEFAULT_PRESENCE_LABELS, idle: 'on watch', idleMode: 'posted' };
  const vector = buildPresenceVector({}, {}, labels);
  assert.equal(vector.line, 'on watch');
  assert.equal(vector.mode, 'posted');
  assert.equal(
    presenceLine({
      reflectionPending: false,
      dueIntention: false,
      carriedThought: false,
      openThread: false,
    }),
    'here with you',
  );
});
