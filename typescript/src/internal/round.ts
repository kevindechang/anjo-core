/**
 * CPython-faithful rounding of an IEEE-754 double.
 *
 * JavaScript's common `toFixed` and scale/round/unscale recipes do not match
 * `round(float, ndigits)` at exact binary midpoints. This implementation works
 * on the double's exact rational representation and applies ties-to-even.
 */
export function pyRound(value: number, digits: number): number {
  if (!Number.isInteger(digits)) throw new RangeError('digits must be an integer');
  if (!Number.isFinite(value) || value === 0) return value;

  // No finite double has a non-zero decimal digit beyond this range. These
  // bounds also prevent accidental huge BigInt allocations for hostile input.
  if (digits > 324) return value;
  if (digits < -308) return value < 0 ? -0 : 0;

  const negative = value < 0;
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setFloat64(0, Math.abs(value), false);
  const high = view.getUint32(0, false);
  const low = view.getUint32(4, false);
  const exponentBits = (high >>> 20) & 0x7ff;
  let mantissa = (BigInt(high & 0xfffff) << 32n) | BigInt(low);
  let binaryExponent: number;

  if (exponentBits === 0) {
    binaryExponent = -1074;
  } else {
    mantissa |= 1n << 52n;
    binaryExponent = exponentBits - 1075;
  }

  let numerator: bigint;
  let denominator: bigint;
  if (digits >= 0) {
    const powerOfFive = 5n ** BigInt(digits);
    const powerOfTwo = binaryExponent + digits;
    if (powerOfTwo >= 0) {
      numerator = mantissa * powerOfFive * 2n ** BigInt(powerOfTwo);
      denominator = 1n;
    } else {
      numerator = mantissa * powerOfFive;
      denominator = 2n ** BigInt(-powerOfTwo);
    }
  } else {
    const places = -digits;
    const powerOfTwo = binaryExponent - places;
    if (powerOfTwo >= 0) {
      numerator = mantissa * 2n ** BigInt(powerOfTwo);
      denominator = 5n ** BigInt(places);
    } else {
      numerator = mantissa;
      denominator = (2n ** BigInt(-powerOfTwo)) * (5n ** BigInt(places));
    }
  }

  let rounded = numerator / denominator;
  const twiceRemainder = 2n * (numerator % denominator);
  if (twiceRemainder > denominator || (twiceRemainder === denominator && rounded % 2n !== 0n)) {
    rounded += 1n;
  }

  if (rounded === 0n) return negative ? -0 : 0;
  const magnitude = digits >= 0
    ? Number(`${rounded.toString()}e-${digits}`)
    : Number(`${rounded.toString()}e${-digits}`);
  return negative ? -magnitude : magnitude;
}
