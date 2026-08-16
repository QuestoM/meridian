// One action implementation. MUI owns the accessible button mechanics and the
// Kairos theme owns every visual state; screens import the canonical controls
// through this entrypoint instead of inventing wrappers or framework paths.
export {
  Button,
  ButtonBase,
  IconButton,
} from '@mui/material';
