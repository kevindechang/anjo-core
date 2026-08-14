import assert from 'node:assert/strict';
import test from 'node:test';

import { pyRound } from '../src/internal/round.js';
import {
  rstripPyWhitespace,
  splitPyWhitespace,
  stripPyWhitespace,
} from '../src/internal/whitespace.js';

test('pyRound follows CPython ties-to-even for positive and negative decimal places', () => {
  assert.equal(pyRound(-0.15625, 4), -0.1562);
  assert.equal(pyRound(0.5625, 3), 0.562);
  assert.equal(pyRound(25, -1), 20);
  assert.equal(pyRound(35, -1), 40);
  assert.equal(pyRound(150, -2), 200);
});

test('pyRound handles IEEE-754 boundaries without overflow artifacts', () => {
  assert.equal(pyRound(1.2345, 400), 1.2345);
  assert.ok(Object.is(pyRound(-1.2345, -400), -0));
  assert.equal(pyRound(Number.POSITIVE_INFINITY, 2), Number.POSITIVE_INFINITY);
  assert.ok(Number.isNaN(pyRound(Number.NaN, 2)));
  assert.throws(() => pyRound(1.2, 1.5), /integer/);
});

test('Python whitespace handling includes separators and NEL but excludes BOM', () => {
  assert.equal(stripPyWhitespace('\u001c\u0085 hello \u3000'), 'hello');
  assert.equal(rstripPyWhitespace('question?\u0085'), 'question?');
  assert.equal(stripPyWhitespace('\ufeffhello\ufeff'), '\ufeffhello\ufeff');
  assert.deepEqual(splitPyWhitespace(' one\u001ctwo\u0085three '), ['one', 'two', 'three']);
});
