#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: convert_result_for_eval.py
"""

import sys
import json
import re
import collections


def convert_result_for_eval(sample_file, result_file, output_file):
    """
    convert_result_for_eval
    """
    sample_list = [line.strip() for line in open(sample_file, 'r',encoding='utf-8')]
    result_list = [line.strip() for line in open(result_file, 'r',encoding='utf-8')]

    assert len(sample_list) == len(result_list)
    fout = open(output_file, 'w',encoding='utf-8')
    for i, sample in enumerate(sample_list):
        sample = json.loads(sample, encoding="utf-8", \
                              object_pairs_hook=collections.OrderedDict)
        response = num_formalinzed(sample["response"])
        fout.write(result_list[i] + "\t" + response + "\n")

    fout.close()
def num_formalinzed(str):
	return re.sub('\d+', '<num>', str).lower()


def main():
    """
    main
    """
    convert_result_for_eval(sys.argv[1],
                            sys.argv[2],
                            sys.argv[3])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
