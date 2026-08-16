import { createTheme } from '@mui/material';
import { createFormAndOverlayOverrides } from './theme-form-overrides';
export { ltrCache, rtlCache } from './emotion-cache';

export const studioPalette = Object.freeze({
  canvas: '#f2eee4',
  surface: '#fbf8f0',
  surfaceMuted: '#eae4d7',
  surfaceRaised: '#ded6c7',
  ink: '#1d1b17',
  inkMuted: '#5d574d',
  inkSubtle: '#70695e',
  line: '#d0c7b7',
  lineStrong: '#8f8572',
  chrome: '#1d1e1a',
  chromeHover: '#31312b',
  accent: '#526d62',
  accentStrong: '#344f47',
  positive: '#376b50',
  positiveSoft: '#dfeadf',
  warning: '#8c5b18',
  warningSoft: '#f1e3c4',
  danger: '#9e3f38',
  dangerSoft: '#f2dcd7',
  info: '#3f6274',
  infoSoft: '#dce7e9',
});

export const studioTypography = Object.freeze({
  hebrew: '"Noto Sans Hebrew Variable", "IBM Plex Sans Hebrew", "IBM Plex Sans", "Arial Hebrew", Arial, sans-serif',
  latin: '"IBM Plex Sans", "Noto Sans Hebrew Variable", "IBM Plex Sans Hebrew", Arial, sans-serif',
  displayHebrew: '"Noto Sans Hebrew Variable", "IBM Plex Sans Hebrew", "IBM Plex Sans", "Arial Hebrew", Arial, sans-serif',
  displayLatin: '"IBM Plex Sans", "Noto Sans Hebrew Variable", "IBM Plex Sans Hebrew", Arial, sans-serif',
  mono: '"IBM Plex Mono", "Courier New", monospace',
});

const focusVisible = {
  outline: `2px solid ${studioPalette.accent}`,
  outlineOffset: 2,
};

const selectedSurface = {
  backgroundColor: studioPalette.surfaceMuted,
  boxShadow: `inset 0 0 0 1px ${studioPalette.lineStrong}`,
};

export function createKairosTheme(requestedDirection) {
  const direction = requestedDirection === 'rtl' ? 'rtl' : 'ltr';
  const fontFamily = direction === 'rtl' ? studioTypography.hebrew : studioTypography.latin;

  return createTheme({
    direction,
    palette: {
      mode: 'light',
      contrastThreshold: 4.5,
      background: {
        default: studioPalette.canvas,
        paper: studioPalette.surface,
      },
      text: {
        primary: studioPalette.ink,
        secondary: studioPalette.inkMuted,
        disabled: studioPalette.inkSubtle,
      },
      primary: {
        light: studioPalette.accent,
        main: studioPalette.accentStrong,
        dark: studioPalette.accentStrong,
        contrastText: studioPalette.surface,
      },
      secondary: {
        light: studioPalette.chromeHover,
        main: studioPalette.chrome,
        dark: studioPalette.chrome,
        contrastText: studioPalette.surface,
      },
      success: {
        light: studioPalette.positiveSoft,
        main: studioPalette.positive,
        dark: studioPalette.positive,
        contrastText: studioPalette.surface,
      },
      warning: {
        light: studioPalette.warningSoft,
        main: studioPalette.warning,
        dark: studioPalette.warning,
        contrastText: studioPalette.surface,
      },
      error: {
        light: studioPalette.dangerSoft,
        main: studioPalette.danger,
        dark: studioPalette.danger,
        contrastText: studioPalette.surface,
      },
      info: {
        light: studioPalette.infoSoft,
        main: studioPalette.info,
        dark: studioPalette.info,
        contrastText: studioPalette.surface,
      },
      divider: studioPalette.line,
      action: {
        active: studioPalette.ink,
        hover: 'rgba(29, 27, 23, 0.06)',
        selected: 'rgba(82, 109, 98, 0.14)',
        disabled: studioPalette.inkSubtle,
        disabledBackground: studioPalette.surfaceMuted,
        focus: 'rgba(82, 109, 98, 0.22)',
      },
    },
    shape: {
      borderRadius: 8,
    },
    shadows: [
      'none',
      '0 8px 24px rgba(58, 49, 36, 0.10)',
      '0 20px 56px rgba(58, 49, 36, 0.16)',
      ...Array(22).fill('0 20px 56px rgba(58, 49, 36, 0.16)'),
    ],
    typography: {
      fontFamily,
      fontSize: 14,
      htmlFontSize: 16,
      body1: { fontSize: 14, lineHeight: '22px', fontWeight: 400 },
      body2: { fontSize: 13, lineHeight: '20px', fontWeight: 500 },
      subtitle1: { fontSize: 16, lineHeight: '24px', fontWeight: 600 },
      subtitle2: { fontSize: 13, lineHeight: '20px', fontWeight: 600 },
      h1: { fontSize: 30, lineHeight: '36px', fontWeight: 600, letterSpacing: '-0.025em' },
      h2: { fontSize: 18, lineHeight: '24px', fontWeight: 600, letterSpacing: '-0.015em' },
      h3: { fontSize: 16, lineHeight: '24px', fontWeight: 600 },
      caption: { fontSize: 12, lineHeight: '18px', fontWeight: 500 },
      overline: { fontSize: 12, lineHeight: '16px', fontWeight: 500, letterSpacing: '0.02em' },
      button: {
        fontSize: 13,
        lineHeight: '20px',
        textTransform: 'none',
        letterSpacing: 0,
        fontWeight: 600,
      },
    },
    transitions: {
      duration: {
        shortest: 110,
        shorter: 110,
        short: 180,
        standard: 180,
        complex: 260,
        enteringScreen: 260,
        leavingScreen: 110,
      },
      easing: {
        easeInOut: 'cubic-bezier(0.2, 0.75, 0.25, 1)',
        easeOut: 'cubic-bezier(0.2, 0.75, 0.25, 1)',
        easeIn: 'cubic-bezier(0.4, 0, 1, 1)',
        sharp: 'cubic-bezier(0.4, 0, 1, 1)',
      },
    },
    components: {
      ...createFormAndOverlayOverrides(studioPalette),
      MuiCssBaseline: {
        styleOverrides: {
          html: {
            backgroundColor: studioPalette.canvas,
            textRendering: 'optimizeLegibility',
          },
          body: {
            backgroundColor: studioPalette.canvas,
            color: studioPalette.ink,
            fontFamily,
            fontVariantNumeric: 'tabular-nums lining-nums',
          },
          '::selection': {
            backgroundColor: studioPalette.accentStrong,
            color: studioPalette.chrome,
          },
          ':focus-visible': focusVisible,
          '@media (prefers-reduced-motion: reduce)': {
            '*, *::before, *::after': {
              animationDuration: '0.01ms',
              animationIterationCount: 1,
              scrollBehavior: 'auto',
              transitionDuration: '0.01ms',
            },
          },
        },
      },
      MuiButtonBase: {
        defaultProps: { disableRipple: true },
      },
      MuiButton: {
        defaultProps: { disableElevation: true },
        styleOverrides: {
          root: {
            minWidth: 44,
            minHeight: 44,
            borderRadius: 8,
            paddingInline: 16,
            paddingBlock: 10,
            fontSize: 13,
            lineHeight: '20px',
            boxShadow: 'none',
            transition: 'background-color 110ms cubic-bezier(0.2, 0.75, 0.25, 1), border-color 110ms cubic-bezier(0.2, 0.75, 0.25, 1)',
            '&.Mui-focusVisible': focusVisible,
          },
          containedPrimary: {
            backgroundColor: studioPalette.accentStrong,
            color: studioPalette.surface,
            '&:hover': { backgroundColor: studioPalette.accent, boxShadow: 'inset 0 0 0 1px rgba(255, 252, 245, 0.30)' },
          },
          outlined: {
            borderColor: studioPalette.lineStrong,
            color: studioPalette.ink,
            '&:hover': { borderColor: studioPalette.accent, backgroundColor: studioPalette.surfaceMuted },
          },
          text: {
            color: studioPalette.inkMuted,
            '&:hover': { backgroundColor: studioPalette.surfaceMuted },
          },
        },
      },
      MuiIconButton: {
        styleOverrides: {
          root: {
            width: 44,
            height: 44,
            borderRadius: 8,
            color: studioPalette.ink,
            '&:hover': { backgroundColor: studioPalette.surfaceMuted },
            '&.Mui-focusVisible': focusVisible,
          },
        },
      },
      MuiCheckbox: {
        styleOverrides: { root: { width: 44, height: 44, color: studioPalette.inkMuted, '&.Mui-focusVisible': focusVisible } },
      },
      MuiRadio: {
        styleOverrides: { root: { width: 44, height: 44, color: studioPalette.inkMuted, '&.Mui-focusVisible': focusVisible } },
      },
      MuiTabs: {
        defaultProps: { slotProps: { indicator: { style: { display: 'none' } } } },
        styleOverrides: { root: { minHeight: 44 } },
      },
      MuiTab: {
        styleOverrides: {
          root: {
            minHeight: 44,
            borderRadius: 8,
            color: studioPalette.inkMuted,
            textTransform: 'none',
            '&.Mui-selected': { color: studioPalette.ink, ...selectedSurface },
            '&.Mui-focusVisible': focusVisible,
          },
        },
      },
      MuiToggleButton: {
        styleOverrides: {
          root: {
            minHeight: 44,
            borderColor: studioPalette.lineStrong,
            borderRadius: 8,
            color: studioPalette.ink,
            textTransform: 'none',
            '&.Mui-selected': selectedSurface,
            '&.Mui-focusVisible': focusVisible,
          },
        },
      },
      MuiMenuItem: {
        styleOverrides: {
          root: {
            minHeight: 44,
            borderRadius: 8,
            fontSize: 13,
            '&.Mui-selected': selectedSurface,
            '&.Mui-focusVisible': focusVisible,
          },
        },
      },
      MuiListItemButton: {
        styleOverrides: {
          root: {
            minHeight: 44,
            borderRadius: 8,
            '&.Mui-selected': selectedSurface,
            '&.Mui-focusVisible': focusVisible,
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: { backgroundImage: 'none', color: studioPalette.ink },
          rounded: { borderRadius: 12 },
        },
      },
      MuiDialog: {
        styleOverrides: {
          paper: {
            border: `1px solid ${studioPalette.line}`,
            borderRadius: 16,
            backgroundColor: studioPalette.surfaceRaised,
            boxShadow: '0 20px 56px rgba(58, 49, 36, 0.16)',
          },
        },
      },
      MuiBackdrop: {
        styleOverrides: { root: { backgroundColor: 'rgba(29, 27, 23, 0.68)' } },
      },
      MuiDataGrid: {
        styleOverrides: {
          root: {
            border: 0,
            borderRadius: 0,
            backgroundColor: studioPalette.surface,
            color: studioPalette.ink,
            fontSize: 12,
            lineHeight: '18px',
          },
          columnHeaders: {
            minHeight: 48,
            backgroundColor: studioPalette.surfaceMuted,
            color: studioPalette.inkMuted,
            fontSize: 12,
            fontWeight: 600,
          },
          columnHeader: {
            '&:focus-visible': focusVisible,
            '&:focus-within': { outlineOffset: -3 },
          },
          cell: {
            minHeight: 48,
            borderColor: studioPalette.line,
            '&:focus-visible': focusVisible,
            '&:focus-within': { outlineOffset: -3 },
          },
          row: {
            minHeight: 48,
            '&:hover': { backgroundColor: studioPalette.surfaceRaised },
            '&.Mui-selected': selectedSurface,
            '&.Mui-selected:hover': selectedSurface,
          },
          footerContainer: {
            minHeight: 48,
            borderColor: studioPalette.line,
          },
        },
      },
      MuiTablePagination: {
        styleOverrides: {
          root: {
            '& .MuiTablePagination-input, & .MuiTablePagination-select': {
              boxSizing: 'border-box',
              minHeight: 44,
            },
            '&& .MuiSelect-select.MuiTablePagination-select': {
              display: 'inline-flex',
              alignItems: 'center',
              minHeight: 44,
            },
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: { minHeight: 48, borderColor: studioPalette.line, color: studioPalette.ink },
          head: { backgroundColor: studioPalette.surfaceMuted, color: studioPalette.inkMuted, fontSize: 12, fontWeight: 600 },
        },
      },
      MuiTooltip: {
        defaultProps: {
          arrow: true,
          slotProps: { tooltip: { dir: direction } },
        },
        styleOverrides: {
          tooltip: {
            maxWidth: '34ch',
            borderRadius: 8,
            backgroundColor: studioPalette.chrome,
            color: studioPalette.surface,
            fontSize: 12,
            lineHeight: '18px',
          },
          arrow: { color: studioPalette.chrome },
        },
      },
      MuiPopover: {
        defaultProps: { slotProps: { paper: { dir: direction } } },
        styleOverrides: {
          paper: { border: `1px solid ${studioPalette.line}`, borderRadius: 12, backgroundColor: studioPalette.surfaceRaised },
        },
      },
      MuiMenu: {
        defaultProps: { slotProps: { paper: { dir: direction } } },
      },
      MuiSelect: {
        defaultProps: { MenuProps: { slotProps: { paper: { dir: direction } } } },
      },
      MuiSkeleton: {
        defaultProps: { animation: 'pulse' },
        styleOverrides: { root: { backgroundColor: studioPalette.surfaceMuted } },
      },
      MuiLinearProgress: {
        styleOverrides: {
          root: { borderRadius: 8, backgroundColor: studioPalette.surfaceMuted },
          bar: { borderRadius: 8, backgroundColor: studioPalette.info },
        },
      },
    },
  });
}
