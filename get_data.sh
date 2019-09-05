"""
#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""

mkdir data

# WMT16 EN-RO
cd data
mkdir wmt16.en-ro
cd wmt16.en-ro
gdown https://drive.google.com/uc?id=1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl
tar -zxvf wmt16.tar.gz
mv wmt16/en-ro/train/corpus.bpe.en train.en
mv wmt16/en-ro/train/corpus.bpe.ro train.ro
mv wmt16/en-ro/dev/dev.bpe.en valid.en
mv wmt16/en-ro/dev/dev.bpe.ro valid.ro
mv wmt16/en-ro/test/test.bpe.en test.en
mv wmt16/en-ro/test/test.bpe.ro test.ro
rm wmt16.tar.gz
rm -r wmt16
cd ../..
python preprocess.py --source-lang en --target-lang ro --trainpref data/wmt16.en-ro/train --validpref data/wmt16.en-ro/valid --testpref data/wmt16.en-ro/test --destdir data-bin/wmt16.en-ro --joined-dictionary --workers 8 --nwordssrc 40000 --nwordstgt 40000
python preprocess.py --source-lang ro --target-lang en --trainpref data/wmt16.en-ro/train --validpref data/wmt16.en-ro/valid --testpref data/wmt16.en-ro/test --destdir data-bin/wmt16.ro-en --joined-dictionary --workers 8 --nwordssrc 40000 --nwordstgt 40000

# WMT14 EN-DE
cd data
mkdir wmt14.en-de
cd wmt14.en-de
mkdir wmt16_en_de
cd wmt16_en_de
gdown https://drive.google.com/uc?id=0B_bZck-ksdkpM25jRUN2X2UxMm8
tar -zxvf wmt16_en_de.tar.gz
cd ..
cp wmt16_en_de/train.tok.clean.bpe.32000.en train.en
cp wmt16_en_de/train.tok.clean.bpe.32000.de train.de
cp wmt16_en_de/newstest2013.tok.bpe.32000.en valid.en
cp wmt16_en_de/newstest2013.tok.bpe.32000.de valid.de
cp wmt16_en_de/newstest2014.tok.bpe.32000.en test.en
cp wmt16_en_de/newstest2014.tok.bpe.32000.de test.de
rm -r wmt16_en_de
cd ../..
python preprocess.py --source-lang en --target-lang de --trainpref data/wmt14.en-de/train --validpref data/wmt14.en-de/valid --testpref data/wmt14.en-de/test --destdir data-bin/wmt14.en-de --joined-dictionary --workers 8 --nwordssrc 32768 --nwordstgt 32768
python preprocess.py --source-lang de --target-lang en --trainpref data/wmt14.en-de/train --validpref data/wmt14.en-de/valid --testpref data/wmt14.en-de/test --destdir data-bin/wmt14.de-en --joined-dictionary --workers 8 --nwordssrc 32768 --nwordstgt 32768

