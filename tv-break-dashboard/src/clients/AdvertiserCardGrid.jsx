import React from 'react';
import { pageText } from './advertisers-helpers';
import { partitionByName } from './advertiser-name-helpers';
import AdvertiserStatCard from './AdvertiserStatCard';

// The advertiser card grid. In the by-name view (grouped) the named records
// render first and every unnamed raw-token record is grouped last under a quiet
// header, so real names never interleave with seed keys. Rows arrive already
// sorted; partitioning preserves their order. In any other sort the grid stays
// flat so the chosen order is not fought.
function AdvertiserCardGrid({ rows, locale, grouped, onOpen }) {
  const renderGrid = (list) => (
    <div className="amz-grid">
      {list.map((row) => (
        <AdvertiserStatCard key={row.advertiser_id} row={row} locale={locale} onOpen={onOpen} />
      ))}
    </div>
  );

  if (!grouped) {
    return renderGrid(rows);
  }

  const { named, unnamed } = partitionByName(rows);
  return (
    <>
      {named.length > 0 && renderGrid(named)}
      {unnamed.length > 0 && named.length > 0 && (
        <div className="amz-group-head">
          {unnamed.length === 1
            ? pageText(locale, 'One pricing row bound to no advertiser, so it prices nothing', 'שורת תמחור אחת שאינה קשורה לאף מפרסם, ולכן אינה מתמחרת דבר')
            : pageText(locale, `${unnamed.length} pricing rows bound to no advertiser, so they price nothing`, `${unnamed.length} שורות תמחור שאינן קשורות לאף מפרסם, ולכן אינן מתמחרות דבר`)}
        </div>
      )}
      {unnamed.length > 0 && renderGrid(unnamed)}
    </>
  );
}

export default AdvertiserCardGrid;
