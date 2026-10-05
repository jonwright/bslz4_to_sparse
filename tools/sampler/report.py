"""report.py OUT [top]: functions by share of the samples in OUT.pcs, using
OUT.maps; symbols via addr2line for the files that hold the samples."""
import sys, collections, subprocess, bisect
out = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
maps = []
for l in open(out + '.maps'):
    p = l.split()
    if len(p) < 6 or 'x' not in p[1]: continue
    a, b = (int(x, 16) for x in p[0].split('-'))
    maps.append((a, b, int(p[2], 16), p[5]))
maps.sort()
base = {}
for l in open(out + '.maps'):
    p = l.split()
    if len(p) >= 6:
        a = int(p[0].split('-')[0], 16)
        if p[5] not in base or a < base[p[5]]: base[p[5]] = a
starts = [m[0] for m in maps]
byfile = collections.defaultdict(collections.Counter); n = 0
for l in open(out + '.pcs'):
    pc = int(l, 16); n += 1
    i = bisect.bisect_right(starts, pc) - 1
    if i >= 0 and pc < maps[i][1]:
        byfile[maps[i][3]][pc - base[maps[i][3]]] += 1
    else:
        byfile['?'][0] += 1
funcs = collections.Counter()
for f, c in byfile.items():
    short = f.split('/')[-1]
    if f.startswith('/') and ('.so' in f or f.endswith('python3') or 'python3.' in f) and sum(c.values()) > n * 0.005:
        addrs = list(c)
        res = subprocess.run(['addr2line', '-f', '-e', f] + ['%x' % a for a in addrs], capture_output=True, text=True).stdout.split('\n')
        for k, a in enumerate(addrs):
            name = res[2 * k] if 2 * k < len(res) else '??'
            funcs['%s (%s)' % (name if name != '??' else '?', short)] += c[a]
    else:
        funcs['(%s)' % short] += sum(c.values())
print('%d samples' % n)
for k, v in funcs.most_common(top):
    print('  %5.1f%%  %s' % (100 * v / n, k[:100]))
