#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape politifact with bs4
"""

import csv
import sys
from dateutil.parser import parse as parse_datetime
from bs4 import BeautifulSoup

def do_command(args):
    soup = BeautifulSoup(args.input, 'html.parser')
    writer = csv.writer(args.output, delimiter='\t')

    writer.writerow(["date", "source", "claim", "judgement", "summary", "url",])
    for id_, stmt in enumerate(soup.find_all('div', class_='statement')):
        date = parse_datetime(stmt.find("span", class_='article__meta').text[3:]).date()
        source = stmt.find("div", class_='statement__source').text.strip()
        claim = stmt.find("p", class_='statement__text').text.strip()
        judgement = stmt.find("div", class_='meter').find('img')['alt']
        summary = stmt.find("div", class_='meter').find('p').text.strip()
        url = "http://politifact.com" + stmt.find("p", class_='statement__text').find('a')["href"]

        writer.writerow([date, source, claim, judgement, summary, url])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', type=argparse.FileType('r'), default=sys.stdin, help="")
    parser.add_argument('--output', type=argparse.FileType('w'), default=sys.stdout, help="")
    parser.set_defaults(func=do_command)

    #subparsers = parser.add_subparsers()
    #command_parser = subparsers.add_parser('command', help='' )
    #command_parser.set_defaults(func=do_command)

    ARGS = parser.parse_args()
    ARGS.func(ARGS)
