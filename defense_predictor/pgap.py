import urllib.parse
from pathlib import Path

import pandas as pd


GENETIC_CODE_11 = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}


_COMPLEMENT = str.maketrans('ACGTNacgtn', 'TGCANtgcan')


def reverse_complement(dna):
    return dna.translate(_COMPLEMENT)[::-1]


def translate_cds(dna):
    dna = dna.upper()
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i + 3]
        aa = GENETIC_CODE_11.get(codon, 'X')
        if aa == '*':
            break
        protein.append(aa)
    if protein:
        # transl_table=11: start codon always translated as M regardless of actual codon.
        protein[0] = 'M'
    return ''.join(protein)


def _parse_attributes(attrs_str):
    out = {}
    for part in attrs_str.split(';'):
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        out[k.strip()] = urllib.parse.unquote(v.strip())
    return out


def parse_pgap_gff(gff_file):
    """Parse a PGAP GFF3 file with embedded genomic FASTA.

    Returns:
        cds_records: list of dicts (one per unique CDS ID), with keys
            id, locus_tag, seqid, strand, product, protein_id, exception,
            segments (list of (start, end) tuples), start (min), end (max).
            CDS lines sharing an ID (programmed frameshifts) are merged.
            Pseudogenes (pseudo=true) are skipped.
        contig_seqs: dict mapping contig_id -> genomic sequence string.
    """
    cds_by_id = {}
    id_order = []
    contig_seqs = {}
    in_fasta = False
    current_contig = None
    current_seq_parts = []

    with open(gff_file) as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            if in_fasta:
                if line.startswith('>'):
                    if current_contig is not None:
                        contig_seqs[current_contig] = ''.join(current_seq_parts)
                    current_contig = line[1:].split()[0]
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)
                continue
            stripped = line.lstrip('#').strip()
            if line.startswith('##FASTA') or line.startswith('## FASTA') or stripped == 'FASTA':
                in_fasta = True
                continue
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 9:
                continue
            seqid, _source, ftype, start, end, _score, strand, _phase, attrs_str = fields[:9]
            if ftype != 'CDS':
                continue
            attrs = _parse_attributes(attrs_str)
            if attrs.get('pseudo', '').lower() == 'true':
                continue
            cds_id = attrs.get('ID')
            if cds_id is None:
                continue
            start_i, end_i = int(start), int(end)
            if cds_id not in cds_by_id:
                cds_by_id[cds_id] = {
                    'id': cds_id,
                    'locus_tag': attrs.get('locus_tag', cds_id),
                    'seqid': seqid,
                    'strand': strand,
                    'product': attrs.get('product', ''),
                    'protein_id': attrs.get('protein_id', ''),
                    'exception': attrs.get('exception', ''),
                    'segments': [],
                }
                id_order.append(cds_id)
            cds_by_id[cds_id]['segments'].append((start_i, end_i))

    if current_contig is not None:
        contig_seqs[current_contig] = ''.join(current_seq_parts)

    records = []
    for cds_id in id_order:
        rec = cds_by_id[cds_id]
        starts = [s for s, _ in rec['segments']]
        ends = [e for _, e in rec['segments']]
        rec['start'] = min(starts)
        rec['end'] = max(ends)
        records.append(rec)
    return records, contig_seqs


def build_feature_df(cds_records):
    """Build a feature DataFrame matching the schema expected by the downstream pipeline."""
    rows = []
    for rec in cds_records:
        rows.append({
            '# feature': 'CDS',
            'product_accession': rec['locus_tag'],
            'genomic_accession': rec['seqid'],
            'start': rec['start'],
            'end': rec['end'],
            'strand': rec['strand'],
            'attributes': '',
        })
    df = pd.DataFrame(rows).sort_values(['genomic_accession', 'start']).reset_index(drop=True)
    df['protein_context_id'] = (df['product_accession'].astype(str) + '|' +
                                df['genomic_accession'].astype(str) + '|' +
                                df['start'].astype(str) + '|' +
                                df['strand'])
    return df


def build_cds_seq_df(cds_records, contig_seqs):
    """Extract CDS nucleotide sequences from the embedded genomic FASTA.

    For plus-strand CDS, segments are concatenated in ascending-start order.
    For minus-strand CDS, each segment is reverse-complemented and segments
    are concatenated in descending-start (mRNA) order.
    """
    rows = []
    for rec in cds_records:
        contig = contig_seqs.get(rec['seqid'])
        if contig is None:
            continue
        segs = sorted(rec['segments'], key=lambda s: s[0])
        if rec['strand'] == '-':
            segs = list(reversed(segs))
        parts = []
        for s, e in segs:
            genomic = contig[s - 1:e]
            if rec['strand'] == '-':
                genomic = reverse_complement(genomic)
            parts.append(genomic)
        seq = ''.join(parts)
        locus_tag = rec['locus_tag']
        pcid = f"{locus_tag}|{rec['seqid']}|{rec['start']}|{rec['strand']}"
        rows.append({
            'protein_context_id': pcid,
            'locus_tag': locus_tag,
            'seq': seq,
        })
    return pd.DataFrame(rows)


def prepare_pgap_inputs(gff_file, workdir):
    """Parse a PGAP GFF and produce the artifacts needed by defense_predictor.

    Returns:
        feature_df: feature table DataFrame (same schema as get_feature_df output)
        cds_seq_df: DataFrame with protein_context_id, locus_tag, seq (CDS nt)
        len_df: DataFrame with product_accession, len (protein aa length)
        faa_path: path to the written protein FASTA (lives in workdir)
    """
    cds_records, contig_seqs = parse_pgap_gff(gff_file)
    feature_df = build_feature_df(cds_records)
    cds_seq_df = build_cds_seq_df(cds_records, contig_seqs)

    proteins = cds_seq_df['seq'].apply(translate_cds)
    len_df = (pd.DataFrame({
        'product_accession': cds_seq_df['locus_tag'],
        'len': proteins.str.len(),
    }).drop_duplicates().reset_index(drop=True))

    faa_path = str(Path(workdir) / 'proteins.faa')
    with open(faa_path, 'w') as f:
        for locus_tag, protein in zip(cds_seq_df['locus_tag'], proteins):
            f.write(f">{locus_tag}\n{protein}\n")

    return feature_df, cds_seq_df, len_df, faa_path
