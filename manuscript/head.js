const fs = require('fs');
const d = require('docx');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, ImageRun,
  WidthType, ShadingType, AlignmentType, BorderStyle, LevelFormat, Footer, PageNumber } = d;

const FONT = 'Calibri';
// --- inline markup: **bold**, *italic*
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t), spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...o });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const EQ = t => new Paragraph({ children: [new TextRun({ text: t, font: 'Cambria Math', italics: false })], alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 } });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: 'bul', level: lvl }, children: runs(t), spacing: { after: 60 } });
const N = (t, ref = 'num') => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60 } });
const NOTE = t => new Paragraph({ children: runs(t, { size: 20, color: '7A1F1F' }), shading: { type: ShadingType.CLEAR, fill: 'FBEDEA' }, spacing: { before: 80, after: 160 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'B03A2E', space: 6 } } });
const CAP = t => new Paragraph({ children: runs(t, { size: 19 }), spacing: { before: 160, after: 80 }, keepNext: true });

const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0,
      children: r.map((c, j) => new TableCell({
        borders, width: { size: widths[j], type: WidthType.DXA },
        shading: i === 0 ? { type: ShadingType.CLEAR, fill: 'E8EDF3' } : undefined,
        margins: { top: 50, bottom: 50, left: 90, right: 90 },
        children: [new Paragraph({ alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
          children: runs(String(c), { size: 19, bold: i === 0 }) })]
      }))
    }))
  });
}

