#!/usr/bin/env python3
"""Assemble the bench page: template.html + solver_shaders.js +
solver_core.js -> ../index.html (deployable) and page_v5.js."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
page = open(os.path.join(HERE, 'template.html')).read()
shaders = open(os.path.join(HERE, 'solver_shaders.js')).read()
core = open(os.path.join(HERE, 'solver_core.js')).read()
assert "const PAGE_VERSION = 'v4';" in page
page = page.replace("const PAGE_VERSION = 'v4';", "const PAGE_VERSION = 'v5.7';")
marker = '// ---------------------------------------------------------------------------\n// Manual job handlers'
assert marker in page
page = page.replace(marker, shaders + '\n' + core + '\n' + marker)
old = """    } else if (job.type === 'custom') {
      rec = await coreCustom(job);
    } else {"""
new = """    } else if (job.type === 'custom') {
      rec = await coreCustom(job);
    } else if (job.type === 'solver') {
      rec = await coreSolver(job);
    } else {"""
assert old in page
page = page.replace(old, new)
open(os.path.join(HERE, '..', 'index.html'), 'w').write(page)
script = re.search(r'<script>(.*)</script>', page, re.S).group(1)
open(os.path.join(HERE, 'page_v5.js'), 'w').write(script)
print('assembled', len(page), 'bytes')
