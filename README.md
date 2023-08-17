# TTS-Bahasa

Text to speech bahasa Indonesia using VITS and Meta MMS.  
Although it is intended for Bahasa Indonesia, it can be used for other languages as well. 
Just changes language code in the python file.


# Installation

## Install the requirements

```bash
pip install -r requirements.txt
```
## Install VITS
  
  ```bash
  git clone clone https://github.com/jaywalnut310/vits.git
  cd vits
  cd monlitic_align
  python3 setup.py build_ext --inplace
  cd ..
  
  ```

# Usage

```bash
python3 mms_tts_ind.py --text "Selamat datang di Indonesia"
```
Supported arguments:
- `-t`, `--text`: text to synthesize (required)
- `-s`, `--save`: save output to file (default: `False`)
- `-o`, `--output`: output filename if saved (default: `out.wav`)  


# Demo
https://bagustris.github.io/tts-bahasa

# References  
- https://github.com/jaywalnut310/vits.git
- https://github.com/facebookresearch/fairseq/blob/main/examples/mms/README.md