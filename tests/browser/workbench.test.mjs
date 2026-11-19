import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test, { after, before } from 'node:test';
import { chromium } from 'playwright-core';

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

let browser;
before(async () => {
  browser = await chromium.launch({ channel: 'chrome', timeout: 60_000 });
}, { timeout: 65_000 });
after(async () => { await browser?.close(); });

async function openWorkbench(page) {
  // Observe each button handler's completion so stale response assertions run after rendering.
  await page.addInitScript(() => {
    const addListener = EventTarget.prototype.addEventListener;
    window.runActions = [];
    EventTarget.prototype.addEventListener = function (type, listener, options) {
      if (type === 'click' && this.id === 'btn-run') {
        addListener.call(this, type, event => {
          window.runActions.push(listener.call(this, event));
        }, options);
      } else {
        addListener.call(this, type, listener, options);
      }
    };
  });
  await page.goto(process.env.SIMVERSE_TEST_URL + '/?testhook=1');
  await page.waitForFunction(() => Boolean(window.__wb?.state.preset));
}

for (const failOld of [false, true]) {
  test(failOld ? 'ignores a stale HTTP failure' : 'keeps the newer run after a stale success',
    { timeout: 25_000 }, async () => {
      const page = await browser.newPage();
      try {
        const requests = [deferred(), deferred()];
        let index = 0;
        await page.route('**/api/v2/run', async route => {
          const next = requests[index++];
          const response = await route.fetch();
          const bundle = await response.json();
          const release = deferred();
          next.resolve({ bundle, release });
          const rejectRequest = await release.promise;
          if (rejectRequest) await route.fulfill({ status: 500, json: { error: 'stale error' } });
          else await route.fulfill({ response });
        });

        await openWorkbench(page);
        await page.locator('#btn-run').click();
        const old = await requests[0].promise;
        assert.equal(old.bundle.manifest.status, 'completed');
        assert.equal(await page.locator('#btn-run').isEnabled(), true);

        await page.locator('#sel-acc').selectOption('0.9');
        await page.locator('#btn-run').click();
        const latest = await requests[1].promise;
        assert.equal(latest.bundle.manifest.status, 'completed');
        const latestId = latest.bundle.manifest.run_id;
        assert.notEqual(old.bundle.manifest.run_id, latestId);

        latest.release.resolve(false);
        await page.evaluate(() => window.runActions[1]);
        const acceptedStatus = await page.locator('#run-status').textContent();
        assert.ok(acceptedStatus.includes(latestId));
        assert.match(await page.locator('#tl-summary').textContent(), /已计价阶段下界/);
        const timeline = await page.locator('#timeline').textContent();
        assert.match(timeline, /commit_rewind（未计价）/);
        assert.match(timeline, /hidden_extract（未计价）/);

        old.release.resolve(failOld);
        await page.evaluate(() => window.runActions[0]);
        assert.equal(await page.locator('#run-status').textContent(), acceptedStatus);
        assert.equal(await page.evaluate(() => window.__wb.state.bundle.manifest.run_id), latestId);
        const downloadEvent = page.waitForEvent('download');
        await page.locator('#btn-export-json').click();
        const download = await downloadEvent;
        assert.equal(download.suggestedFilename(), 'simverse-run-' + latestId.slice(7, 19) + '.json');
        const exported = JSON.parse(await readFile(await download.path(), 'utf8'));
        assert.equal(exported.manifest.run_id, latestId);
      } finally {
        await page.close();
      }
    });
}

test('keeps the current run status and labels partial Engram timing after a stale CSV fails',
  { timeout: 25_000 }, async () => {
    const page = await browser.newPage();
    const csvStarted = deferred();
    const releaseCsv = deferred();
    try {
      await page.route('**/api/v2/export/ops', async route => {
        csvStarted.resolve(route.request().postDataJSON().manifest.run_id);
        await releaseCsv.promise;
        await route.fulfill({ status: 500, json: { error: 'stale export error' } });
      });
      await openWorkbench(page);
      await page.locator('#btn-run').click();
      await page.evaluate(() => window.runActions[0]);
      const status = page.locator('#run-status');
      const oldStatus = await status.textContent();
      await page.locator('#btn-export-csv').click();
      assert.ok(oldStatus.includes(await csvStarted.promise));

      await page.locator('#sel-model').selectOption('deepseek_v41_flash');
      await page.locator('#btn-run').click();
      await page.evaluate(() => window.runActions[1]);
      const currentStatus = await status.textContent();
      assert.match(currentStatus, /run_id sha256:/);
      assert.notEqual(currentStatus, oldStatus);
      const timing = page.locator('#tl-summary');
      assert.match(await timing.textContent(), /已计价阶段下界/);
      assert.match(await timing.getAttribute('title'), /Engram host/);

      releaseCsv.resolve();
      const exportStatus = page.locator('#export-status');
      await exportStatus.waitFor({ state: 'visible' });
      assert.match(await exportStatus.textContent(), /CSV 导出失败/);
      assert.equal(await status.textContent(), currentStatus);
    } finally {
      releaseCsv.resolve();
      await page.close();
    }
  });
