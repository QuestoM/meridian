import React from 'react';

// An engine field name printed in a narrow column.
//
// To a line breaker min_retention_floor is one unbreakable word: LOW LINE
// carries no break opportunity in UAX 14, so the name either fits its column or,
// under overflow-wrap: anywhere, breaks mid-word. Measured in a 420 px dock
// before this existed: the proposal diff's field cell was 98 px, the name is
// wider, and the row that says which variable is about to change read
// "min_retention_flo" over "or".
//
// The column is wider now, and this adds an explicit break opportunity after
// each underscore so a name too long for any width still breaks where a reader
// would break it. <wbr> adds no character: the text copied out of the cell is
// the field name exactly as the engine spells it.
export default function FieldName({ name }) {
  const text = String(name ?? '');
  const parts = text.split('_');
  return (
    <>
      {parts.map((part, index) => {
        const last = index === parts.length - 1;
        return (
          <React.Fragment key={index}>
            {last ? part : `${part}_`}
            {last ? null : <wbr />}
          </React.Fragment>
        );
      })}
    </>
  );
}
