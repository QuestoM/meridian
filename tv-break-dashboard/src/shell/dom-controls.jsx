import React from 'react';

export function cx(...values) {
  return values.flat().filter(Boolean).join(' ');
}

// Structural pressables and specialised native inputs deliberately live in a
// dependency-light module. Timeline bands, rows, password fields and range
// controls keep their native semantics without pulling MUI into SSR/readout
// harnesses or inventing screen-local tags.
export const Pressable = React.forwardRef(function Pressable({
  type = 'button', loading = false, disabled, className = '', children, ...rest
}, ref) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx('studio-pressable', className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...rest}
    >
      {children}
    </button>
  );
});

export const InputControl = React.forwardRef(function InputControl({ className = '', invalid, ...rest }, ref) {
  return <input ref={ref} className={cx('studio-input-control', className)} aria-invalid={invalid || undefined} {...rest} />;
});

export const SelectControl = React.forwardRef(function SelectControl({ className = '', invalid, children, ...rest }, ref) {
  return <select ref={ref} className={cx('studio-select-control', className)} aria-invalid={invalid || undefined} {...rest}>{children}</select>;
});

export const TextAreaControl = React.forwardRef(function TextAreaControl({ className = '', invalid, children, ...rest }, ref) {
  return <textarea ref={ref} className={cx('studio-textarea-control', className)} aria-invalid={invalid || undefined} {...rest}>{children}</textarea>;
});
