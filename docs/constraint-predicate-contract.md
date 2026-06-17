# Constraint predicate contract

This document defines the exact frozen JSON shape for the `where` predicate on a
placement constraint. The frontend and the backend are both built to this shape.
Do not rename fields, operators, or combinators without a coordinated migration.

## Shape

```
Group     = { "combinator": "and" | "or", "conditions": [ Node, ... ] }
Node      = Group | Condition
Condition = { "field": <field>, "operator": <op>, "value": <value> }
```

A `conditions` list may mix nested groups and conditions. Nesting is fully
recursive. An empty `conditions` list yields no match (false for both `and` and
`or`).

## Fields, operators, and value types

| field | what it tests | allowed operators | value type |
|---|---|---|---|
| `programme` | programme Title | `is`, `is_not`, `contains`, `not_contains`, `starts_with`, `ends_with`, `regex`, `in` | string; array for `in` |
| `genre` | program_type | same as `programme` | string; array for `in` |
| `daypart` | `morning`, `noon`, `evening`, `prime`, `night` | `is`, `is_not`, `in` | string; array for `in` |
| `weekday` | `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, `Sun` | `is`, `is_not`, `in` | string; array for `in` |
| `date` | ISO yyyy-mm-dd broadcast date | `is`, `before`, `after`, `between`, `in` | ISO date string; `{"min": iso, "max": iso}` for `between`; array for `in` |
| `hour` | integer 0..23, from segment broadcast time | `eq`, `lt`, `lte`, `gt`, `gte`, `between` | integer; `{"min": int, "max": int}` for `between` |

String comparisons (`is`, `is_not`, `contains`, `not_contains`, `starts_with`,
`ends_with`, `in`) are case-insensitive. `regex` uses Python `re` with
`re.IGNORECASE`. A malformed regex makes that single condition false (no error,
no match-all).

## Channel field

The channel is NOT part of the predicate. Every constraint automatically scopes
to the operator's own channel (read from `KairosSettings.operator_channel`). An
operator can never constrain another channel's breaks.

## Stored shape

A stored constraint is:

```json
{
  "where": <Group>,
  "effect": "fix_offset | offset_window | pin_count | duration_range | gold | forbid",
  "offset_seconds": null,
  "offset_min_seconds": null,
  "offset_max_seconds": null,
  "count": null,
  "duration_seconds": null,
  "duration_min_seconds": null,
  "duration_max_seconds": null,
  "order_index": null,
  "notes": ""
}
```

`where` is optional. When absent the legacy flat `scope_type` / `scope_value`
matching is used unchanged, so existing constraints keep working.

## Worked example 1: nested AND/OR

Pin the first break at 22 minutes in any prime or evening Drama or Crime segment:

```json
{
  "combinator": "and",
  "conditions": [
    {
      "combinator": "or",
      "conditions": [
        { "field": "genre",   "operator": "is", "value": "Drama" },
        { "field": "genre",   "operator": "is", "value": "Crime" }
      ]
    },
    {
      "combinator": "or",
      "conditions": [
        { "field": "daypart", "operator": "is", "value": "prime" },
        { "field": "daypart", "operator": "is", "value": "evening" }
      ]
    }
  ]
}
```

Full POST body for this constraint:

```json
{
  "scope_type": "always",
  "effect": "fix_offset",
  "offset_seconds": 1320,
  "order_index": 1,
  "notes": "first break at 22 min in prime/evening Drama or Crime",
  "where": {
    "combinator": "and",
    "conditions": [
      {
        "combinator": "or",
        "conditions": [
          { "field": "genre", "operator": "is", "value": "Drama" },
          { "field": "genre", "operator": "is", "value": "Crime" }
        ]
      },
      {
        "combinator": "or",
        "conditions": [
          { "field": "daypart", "operator": "is", "value": "prime" },
          { "field": "daypart", "operator": "is", "value": "evening" }
        ]
      }
    ]
  }
}
```

## Worked example 2: regex programme match

Forbid breaks on any episode whose title starts with "חדשות" (any news edition):

```json
{
  "scope_type": "always",
  "effect": "forbid",
  "notes": "no breaks on any news edition",
  "where": {
    "combinator": "and",
    "conditions": [
      { "field": "programme", "operator": "regex", "value": "^חדשות" }
    ]
  }
}
```

`regex` uses Python `re` with `re.IGNORECASE`. The pattern is matched anywhere
in the title (use `^` to anchor at the start, `$` at the end).
