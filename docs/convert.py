import os
import sys
import glob
from pathlib import Path

import nbconvert
from nbconvert import RSTExporter, MarkdownExporter
from nbconvert.writers import FilesWriter
import nbformat
from traitlets.config import Config


thisdir = os.path.dirname(os.path.abspath(__file__))
codedir = os.path.join(os.path.dirname(thisdir), "example")
tgtdir = os.path.join(thisdir, "_example")

def nb2rst(fname, outfile):
    with open(fname, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
    dirname = os.path.dirname(outfile)
    basename = os.path.basename(outfile)
    name = os.path.splitext(basename)[0]
    rst_exporter = RSTExporter()
    (body, resources) = rst_exporter.from_notebook_node(notebook_content)
    c = Config()
    c.FilesWriter.build_directory = dirname
    files_writer = FilesWriter(config=c)
    files_writer.write(body, resources, name)
    

if __name__ == '__main__':

    filelist = []
    for root, dirname, filenames in os.walk(codedir):
        for fname in filenames:
            ext = os.path.splitext(fname)[-1]
            if ext == ".ipynb":
                filelist.append(os.path.join(root, fname))
    print(f"Number of notebooks to convert: {len(filelist)}")

    pairlist = []
    for fpath in filelist:
        tgt_fpath = fpath.replace(".ipynb", ".rst")
        tgt_fpath = tgt_fpath.replace(codedir, tgtdir)
        pairlist.append((fpath, tgt_fpath))

    # print(pairlist)
    for fpath, tgt_fpath in pairlist:
        nb2rst(fpath, tgt_fpath)