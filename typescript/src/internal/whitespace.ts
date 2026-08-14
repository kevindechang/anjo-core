/** Character class used by CPython's parameterless str.strip/split methods. */
export const PYTHON_WHITESPACE =
  '\\t\\n\\x0b\\f\\r\\x1c-\\x1f \\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';

const TRIM = new RegExp(`^[${PYTHON_WHITESPACE}]+|[${PYTHON_WHITESPACE}]+$`, 'g');
const RTRIM = new RegExp(`[${PYTHON_WHITESPACE}]+$`);
const RUN = new RegExp(`[${PYTHON_WHITESPACE}]+`);

/** Equivalent to Python `str.strip()` for Unicode whitespace. */
export function stripPyWhitespace(value: string): string {
  return value.replace(TRIM, '');
}

/** Equivalent to Python `str.rstrip()` for Unicode whitespace. */
export function rstripPyWhitespace(value: string): string {
  return value.replace(RTRIM, '');
}

/** Equivalent to Python `str.split()` with no separator. */
export function splitPyWhitespace(value: string): string[] {
  const stripped = stripPyWhitespace(value);
  return stripped ? stripped.split(RUN) : [];
}
