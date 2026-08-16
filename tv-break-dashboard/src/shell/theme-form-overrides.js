// Field and side-overlay geometry shared by every MUI-backed workspace. Keeping
// these overrides together prevents feature sheets from repairing the same
// focus, switch and drawer internals with incompatible local selectors.
export function createFormAndOverlayOverrides(palette) {
  return {
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          minHeight: 44,
          borderRadius: 8,
          backgroundColor: palette.surface,
          color: palette.ink,
          fontSize: 13,
          lineHeight: '20px',
          '& .MuiOutlinedInput-notchedOutline': { borderColor: palette.lineStrong },
          '&:hover .MuiOutlinedInput-notchedOutline': { borderColor: palette.inkSubtle },
          '&.Mui-focused': { outline: 'none' },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: palette.accent,
            borderWidth: 1,
          },
        },
        input: {
          minWidth: 0,
          paddingBlock: 11,
          paddingInline: 12,
        },
      },
    },
    MuiInputLabel: {
      styleOverrides: {
        root: {
          color: palette.inkMuted,
          fontSize: 12,
          lineHeight: '18px',
          '&.MuiInputLabel-shrink': {
            marginInline: -4,
            paddingInline: 4,
            backgroundColor: palette.surface,
          },
        },
      },
    },
    MuiFormHelperText: {
      styleOverrides: {
        root: { color: palette.inkMuted, fontSize: 12, lineHeight: '18px' },
      },
    },
    MuiSwitch: {
      styleOverrides: {
        root: {
          minWidth: 44,
          minHeight: 44,
          '&.MuiSwitch-sizeSmall': {
            width: 44,
            height: 44,
            padding: '10px 3px',
            '& .MuiSwitch-switchBase': { top: 10 },
          },
          '&:has(.Mui-focusVisible)': {
            outline: `2px solid ${palette.accent}`,
            outlineOffset: 2,
            borderRadius: 12,
          },
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        root: {
          '& .MuiBackdrop-root': { top: 'var(--shell-header-height)' },
        },
        paper: {
          top: 'var(--shell-header-height)',
          bottom: 0,
          height: 'auto',
          maxHeight: 'calc(100dvh - var(--shell-header-height))',
          borderColor: palette.line,
          backgroundColor: palette.surfaceRaised,
          boxShadow: '0 20px 56px rgba(58, 49, 36, 0.16)',
        },
      },
    },
  };
}
