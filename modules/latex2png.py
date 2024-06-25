# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
import os
import re
import sys
import io
import glob
import tempfile
import shlex
import subprocess
import traceback
from PIL import Image, ImageChops, ImageDraw, ImageFont
import shutil


class Latex:
    BASE = r'''
\documentclass[varwidth]{standalone}
\usepackage{fontspec,unicode-math}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\begin{document}
%s
\end{document}
'''

# \setmathfont{%s}
# \thispagestyle{empty}

    def __init__(self, math, dpi=250, font='Latin Modern Math', tex_type="formula"):
        '''takes list of math code. `returns each element as PNG with DPI=`dpi`'''
        self.math = math
        self.dpi = dpi
        self.font = font
        self.tex_type = tex_type
        self.prefix_line = self.BASE.split("\n").index(
            "%s")  # used for calculate error formula index

    def write(self, return_bytes=False):
        # inline = bool(re.match('^\$[^$]*\$$', self.math)) and False   
        if type(self.math) == str:
            self.math = [self.math]
        for i in range(len(self.math)):
            if self.tex_type == "formula":
                self.math[i] = r'\begin{displaymath}'+self.math[i]+r'\end{displaymath}'
            else:
                self.math[i] = r'\begin{displaymath}\n\text{'+self.math[i]+r'}\n\end{displaymath}'

        try:
            # clear and create temp dir
            temp_dir = os.path.join(os.getcwd(), 'temp')
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.mkdir(temp_dir)
            workdir = tempfile.mkdtemp(dir='temp')
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
            # print(self.BASE % (self.font, self.math))
            with os.fdopen(fd, 'w+') as f:
                # document = self.BASE % (self.font, '\n'.join(self.math))
                document = self.BASE % ('\n'.join(self.math))
                # print(document)
                f.write(document)

            png, error_index = self.convert_file(
                texfile, workdir, return_bytes=return_bytes)
            return png, error_index

        finally:
            pass
            # if os.path.exists(texfile):
            #     try:
            #         os.remove(texfile)
            #     except PermissionError:
            #         pass

    def convert_file(self, infile, workdir, return_bytes=False):
        infile = infile.replace('\\', '/')
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            cmd = 'xelatex -interaction nonstopmode -file-line-error -output-directory %s %s' % (
                workdir.replace('\\', '/'), infile)

            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            sout, serr = p.communicate()
            # print("Complete convert to pdf")
            pdffile = infile.replace('.tex', '.pdf')
            pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))

            cmd = 'convert -density %i -colorspace gray %s -quality 90 %s' % (
                self.dpi,
                pdffile,
                pngfile,
            )  # -bg Transparent -z 9
            if sys.platform == 'win32':
                cmd = 'magick ' + cmd
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            sout, serr = p.communicate()
            # print("Complete convert to png")
            if p.returncode != 0:
                raise Exception('PDFpng error', serr, cmd, os.path.exists(
                    pdffile), os.path.exists(infile))
            if return_bytes:
                if len(self.math) > 1 and type(self.math) == list:
                    png = [open(pngfile.replace('.png', '')+'-%i.png' %
                                i, 'rb').read() for i in range(len(self.math))]
                else:
                    png = [open(pngfile.replace(
                        '.png', '')+'.png', 'rb').read()]
            else:
                # return path
                if len(self.math) > 1 and type(self.math) == list:
                    png = [(pngfile.replace('.png', '')+'-%i.png' % i)
                           for i in range(len(self.math))]
                else:
                    png = [(pngfile.replace('.png', '')+'.png')]
            return png, None
        except Exception as e:
            print(e)
        finally:
            pass
            # Cleanup temporaries
            # basefile = infile.replace('.tex', '')
            # tempext = ['.aux', '.pdf', '.log']
            # if return_bytes:
            #     ims = glob.glob(basefile+'*.png')
            #     for im in ims:
            #         os.remove(im)
            # for te in tempext:
            #     tempfile = basefile + te
            #     if os.path.exists(tempfile):
            #         os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, return_error_index=False, remove_alpha=True, trim=True, **kwargs):
    pngs, error_index = Latex(tex, **kwargs).write(return_bytes=True)
    # images = [Image.open(io.BytesIO(d)) for d in pngs]
    images = []
    for d in pngs:
        if d is not None:
            img_pil = Image.open(io.BytesIO(d))
            if remove_alpha:
                img_pil = img_pil.convert('RGB')
            if trim:
                bg = Image.new(img_pil.mode, img_pil.size, img_pil.getpixel((0, 0)))
                img_pil = img_pil.crop(ImageChops.difference(img_pil, bg).getbbox())
            images.append(img_pil)
        else:
            images.append(None)
    return (images, error_index) if return_error_index else images


def zhtext2pil(zh_string, word_size=18):
    """convert zh-text to pil image"""
    word_num = len(zh_string)
    img = Image.new('RGB', ((word_size-3)*word_num, word_size), 'white')
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("simhei.ttf", 15, encoding="utf-8")
    draw.text((0, 1), zh_string, (0, 0, 0), font=fontText)
    # draw.rectangle([0, 0, img.size[0]-1, img.size[1]-1], fill=None, outline=(255,0,0), width=1)
    return img


def extract(text, expression=None):
    """extract text from text by regular expression

    Args:
        text (str): input text
        expression (str, optional): regular expression. Defaults to None.

    Returns:
        str: extracted text
    """
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text)
        return results, True if len(results) != 0 else False
    except Exception:
        traceback.print_exc()



if __name__ == '__main__':
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        src = r'\int_{-\infty}^\infty e^{-x^2} \, dx = \sqrt{\pi}'

    print('Equation is: %s' % src)
    print(Latex(src).write())
