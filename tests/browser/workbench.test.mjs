import assert from 'node:assert/strict';
import test from 'node:test';
import { chromium } from 'playwright-core';

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
}

for (const failOld of [false, true]) {
  test(failOld ? 'ignores a stale HTTP failure' : 'keeps the newer run after a stale success',
    { timeout: 25_000 }, async () => {
      const browser = await chromium.launch({ channel: 'chrome' });
      try {
        const page = await browser.newPage();
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

        await page.goto(process.env.SIMVERSE_TEST_URL + '/?testhook=1');
        await page.waitForFunction(() => Boolean(window.__wb?.state.preset));
        await page.evaluate(() => { window.older = window.__wb.doRun(); });
        const old = await requests[0].promise;
        assert.equal(old.bundle.manifest.status, 'completed');
        assert.equal(await page.locator('#btn-run').isEnabled(), true);

        await page.locator('#sel-acc').selectOption('0.9');
        await page.evaluate(() => { window.newer = window.__wb.doRun(); });
        const latest = await requests[1].promise;
        assert.equal(latest.bundle.manifest.status, 'completed');
        const latestId = latest.bundle.manifest.run_id;
        assert.notEqual(old.bundle.manifest.run_id, latestId);

        latest.release.resolve(false);
        await page.evaluate(() => window.newer);
        const acceptedStatus = await page.locator('#run-status').textContent();
        assert.ok(acceptedStatus.includes(latestId));
        assert.match(await page.locator('#tl-summary').textContent(), /已计价阶段下界/);
        const timeline = await page.locator('#timeline').textContent();
        assert.match(timeline, /commit_rewind（未计价）/);
        assert.match(timeline, /hidden_extract（未计价）/);

        old.release.resolve(failOld);
        await page.evaluate(() => window.older);
        assert.equal(await page.locator('#run-status').textContent(), acceptedStatus);
        assert.equal(await page.evaluate(() => window.__wb.state.bundle.manifest.run_id), latestId);
        const downloadEvent = page.waitForEvent('download');
        await page.locator('#btn-export-json').click();
        const download = await downloadEvent;
        assert.equal(download.suggestedFilename(), 'simverse-run-' + latestId.slice(7, 19) + '.json');
      } finally {
        await browser.close();
      }
    });
}
