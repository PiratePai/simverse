/* Simverse CSV 转义（RFC 4180）——前端导出与 CLI/测试共用语义。
 * 字段含逗号/引号/换行时加引号包裹，内部引号翻倍；不做字符替换，
 * 保留原始字段内容。Node 测试通过 window.simverseCsv 直接调用。 */
(function (root) {
  'use strict';
  function escapeField(v) {
    var s = (v === null || v === undefined) ? '' : String(v);
    if (/[",\r\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }
  function row(fields) { return fields.map(escapeField).join(','); }
  function toCsv(cols, records) {
    var lines = [row(cols)];
    (records || []).forEach(function (r) {
      lines.push(row(cols.map(function (c) { return r[c]; })));
    });
    return lines.join('\n');
  }
  var api = { escapeField: escapeField, row: row, toCsv: toCsv };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.simverseCsv = api;
})(typeof window !== 'undefined' ? window : globalThis);
