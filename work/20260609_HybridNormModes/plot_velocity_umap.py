#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""1 行 shim: 実体は work/20260609_Hybrid5x3/viz/plot_velocity_umap.py。CLI 引数はそのまま委譲。"""
import os, runpy
runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "20260609_Hybrid5x3", "viz",
                            "plot_velocity_umap.py"), run_name="__main__")
