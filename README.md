# Football Edge

A play-money NFL betting simulator. No real money is involved.

Live site: https://spartanot.github.io/football-edge/

- `index.html` is the app.
- `update.py` downloads the latest nflverse schedule, betting lines, and play-by-play data, rebuilds the prediction model, and writes `data.json`.
- `.github/workflows/update.yml` runs the updater twice a day and publishes the site.

To update right away: Actions tab → Update data → Run workflow.

Data: nflverse (https://github.com/nflverse). Help with problem gambling: 1-800-GAMBLER.
