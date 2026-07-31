import createCache from '@emotion/cache';
import { prefixer } from 'stylis';
import rtlPlugin from '@mui/stylis-plugin-rtl';
import { createTheme } from '@mui/material';

export const ltrCache = createCache({ key: 'mui' });
export const rtlCache = createCache({
  key: 'muirtl',
  stylisPlugins: [prefixer, rtlPlugin],
});

export function createKairosTheme(direction) {
  return createTheme({
    direction,
    palette: {
      mode: 'light',
      background: {
        default: '#f7f8fa',
        paper: '#ffffff',
      },
      text: {
        primary: '#111827',
        secondary: '#5b6573',
      },
      primary: {
        main: '#0d1b2a',
      },
      success: {
        main: '#0f8b7e',
      },
      warning: {
        main: '#b86e00',
      },
      divider: '#dde2e8',
    },
    shape: {
      borderRadius: 6,
    },
    typography: {
      fontFamily:
        'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      button: {
        textTransform: 'none',
        letterSpacing: 0,
        fontWeight: 620,
      },
    },
    components: {
      MuiButton: {
        defaultProps: { disableElevation: true },
        styleOverrides: {
          root: {
            minHeight: 34,
            borderRadius: 6,
            fontSize: 12,
            lineHeight: 1,
            boxShadow: 'none',
          },
        },
      },
      MuiIconButton: {
        styleOverrides: {
          root: {
            width: 34,
            height: 34,
            borderRadius: 6,
            color: '#111827',
          },
        },
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            backgroundColor: '#ffffff',
            fontSize: 12,
          },
          input: {
            paddingTop: 8,
            paddingBottom: 8,
          },
        },
      },
      MuiInputLabel: {
        styleOverrides: {
          root: {
            fontSize: 12,
          },
        },
      },
      MuiDataGrid: {
        styleOverrides: {
          root: {
            border: 0,
            fontSize: 12,
            color: '#111827',
          },
          columnHeaders: {
            backgroundColor: '#fbfcfd',
            color: '#5b6573',
            fontSize: 11,
            fontWeight: 700,
          },
          cell: {
            borderColor: '#dde2e8',
          },
          row: {
            '&:hover': {
              backgroundColor: '#fbfcfd',
            },
          },
        },
      },
      MuiTooltip: {
        defaultProps: {
          // Hebrew tooltips read right-to-left; the popper is portaled outside
          // the rtl shell, so the bubble needs the direction set explicitly.
          slotProps: { tooltip: { dir: direction } },
        },
      },
      // Select/Menu/Popover portal their list to document.body, outside the rtl
      // shell, so without an explicit direction they open left-to-right in Hebrew.
      MuiPopover: {
        defaultProps: { slotProps: { paper: { dir: direction } } },
      },
      MuiMenu: {
        defaultProps: { slotProps: { paper: { dir: direction } } },
      },
      MuiSelect: {
        defaultProps: { MenuProps: { slotProps: { paper: { dir: direction } } } },
      },
    },
  });
}
