# Fonts Bundle

This project vendors a small set of open-licensed fonts so the synthetic CAPTCHA generator
and every style variant work predictably even in offline environments. All files were
retrieved from the public [google/fonts](https://github.com/google/fonts) repository and
retain their original licenses. Variable fonts (`*-Variable.ttf`) expose multiple weights
through a single file and are supported by Pillow.

- Apache 2.0: Open Sans, Roboto, Roboto Mono
- SIL Open Font License 1.1: Lato, Montserrat, Oswald, Poppins, Raleway, Merriweather, Noto Sans, Comic Neue, Dancing Script, Great Vibes
- Ubuntu Font License 1.0: Ubuntu
- Public Domain (Bitstream / DejaVu project): DejaVu Sans

If you add additional fonts, place the `.ttf` files in this directory (renaming to an ASCII
filename if necessary) and update this manifest accordingly. All generator styles will
automatically discover them at runtime.
