PDF=${1:-data/pdfs/ocr_test/ocr_0.pdf}


srun -p s2_bigdata --gres=gpu:1 --async python process_pdf.py --pdf $PDF --vis


rm batchscript*
