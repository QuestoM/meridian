// A completed job is not automatically a routine success. A zero-break result
// is still a real result, but it needs a warning before the plan can be frozen.
export function announceRunResult(owned, notify) {
  const zeroBreaks = Number(owned?.total_breaks) === 0;
  if (zeroBreaks) {
    notify?.(
      'The run finished with zero breaks on your channel. Review the result before freezing it.',
      'ההרצה הסתיימה עם אפס ברייקים בערוץ שלכם. יש לבדוק את התוצאה לפני הקפאה.',
    );
  } else {
    notify?.(
      `The weekly plan was run: ${owned?.total_breaks ?? '-'} breaks on your channel.`,
      `התוכנית השבועית רצה: ${owned?.total_breaks ?? '-'} ברייקים בערוץ שלכם.`,
    );
  }
  return zeroBreaks;
}
