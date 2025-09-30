import joblib, itertools, numpy as np, random

class FedSpaceScheduler:
    """Scheduler implementing FedSpace Offline and Online phases."""

    def __init__(self, window_len, regressor_pkl,
                 connectivity_table: dict[int, list[int]], n_rand=128, n_min=1, n_max=4):
        self.I0 = window_len
        self.rgrs = joblib.load(regressor_pkl)
        self.n_rand = n_rand
        self.n_min, self.n_max = n_min, n_max
        self.pattern_idx = 0
        self.round = 0

        self._connectivity_table = connectivity_table

        assert set(self._connectivity_table.keys()) == set(range(self.I0)), "connectivity_table keys must be exactly 0..I0-1"

        self.K = max((c for slot in connectivity_table.values() for c in slot)) + 1

        self._rng = random.Random(0)
        self._np_rng = np.random.default_rng(0)
        self._nonempty_slots = [s for s in range(self.I0) if len(self._connectivity_table.get(s, [])) > 0]
        if not self._nonempty_slots:
            self._nonempty_slots = list(range(self.I0))
        self._plan_state = None

        self._slot_weights = np.array(
            [len(self._connectivity_table.get(s, [])) for s in range(self.I0)],
            dtype=float
        )
        w = self._slot_weights[self._nonempty_slots]
        self._nonempty_probs = (w / w.sum()) if w.sum() > 0 else None

        self.last_up = [-1] * self.K
        self.agg_ctr = 0
        self.next_a = None
        self.pattern_history = []
        self.start_new_window(T=1.0)
        print(f"[FedSpace] initial pattern idxs = {np.where(self.current_a==1)[0].tolist()}")
        print(f"[FedSpace] planned agg count this window = {int(self.current_a.sum())}")

    def _sample_pattern(self):
        nagg = self._rng.randint(self.n_min, self.n_max)
        if self._nonempty_probs is not None:
            idxs = self._np_rng.choice(
                self._nonempty_slots,
                size=min(nagg, len(self._nonempty_slots)),
                replace=False,
                p=self._nonempty_probs
            ).tolist()
        else:
            idxs = self._rng.sample(self._nonempty_slots, k=min(nagg, len(self._nonempty_slots)))
        a = np.zeros(self.I0, dtype=int); a[idxs] = 1
        return a
    
    def begin_next_plan(self, T):
        # initialize random search for next window
        self._plan_state = {
            "T": float(T),
            "done": 0,
            "best_util": -1e30,
            "best_a": None,
            "last_up_snapshot": self.last_up.copy(),
        }

    def plan_next_step(self, budget=16):
        # evaluate a small number of candidates;
        st = self._plan_state
        if st is None:
            return

        remaining = self.n_rand - st["done"]
        if remaining <= 0:
            # finished: publish result and clear
            self.next_a = st["best_a"]
            self._plan_state = None
            return

        this_batch = min(budget, remaining)
        for _ in range(this_batch):
            a = self._sample_pattern()
            util = self._utility_sum(a, st["T"], st["last_up_snapshot"])
            if util > st["best_util"]:
                st["best_util"] = util
                st["best_a"] = a
        st["done"] += this_batch

        # expose the best-so-far
        self.next_a = st["best_a"]

    def should_aggregate(self, connected_clients, loss_T: float) -> bool:
        if self.current_a is None or self.pattern_idx == self.I0:
            if self.next_a is not None:
                self.current_a = self.next_a
                self.next_a = None
                self.pattern_idx = 0
            else:
                self.start_new_window(loss_T)   # picks range + a (fast; see below)
        do_update = bool(self.current_a[self.pattern_idx])
        self.pattern_idx += 1
        self.round += 1
        return do_update
    
    def record_upload(self, uploaded_clients):
        if not uploaded_clients:
            return
        for k in uploaded_clients:
            if 0 <= k < self.K:
                self.last_up[k] = self.agg_ctr
        self.agg_ctr += 1


    def _pick_pattern(self, loss_T, last_up):
        candidates = []
        if hasattr(self, "current_a") and self.current_a is not None:
            base = self.current_a.copy()
            for _ in range(12):
                a = base.copy()
                # flip one bit off and one on within the allowed range
                ones  = np.where(a == 1)[0].tolist()
                zeros = [z for z in np.where(a == 0)[0].tolist() if z in self._nonempty_slots]
                if ones and zeros:
                    off = self._rng.choice(ones)
                    on = self._rng.choice(zeros)
                    a[off] = 0
                    a[on] = 1
                util = self._utility_sum(a, loss_T, last_up)
                candidates.append((util, a))
        # weighted randoms
        for _ in range(self.n_rand):
            a = self._sample_pattern()
            util = self._utility_sum(a, loss_T, last_up)
            candidates.append((util, a))
        return max(candidates, key=lambda t: t[0])[1]

    def _utility_sum(self, a, loss_T, last_up):
        util = 0.0
        fake_slot = self.round             # drives connectivity lookups
        fake_last = last_up.copy()         # last agg index per client
        fake_agg  = self.agg_ctr           # current agg index
        buffer = set()
        feat_rows = []

        for flag in a:
            C_i = self._connected_in_fake_slot(fake_slot)
            buffer.update(C_i)

            if flag:
                # staleness in units of aggregation rounds
                s_l = [
                    (-1 if k not in buffer else (0 if fake_last[k] < 0 else (fake_agg - fake_last[k] - 1)))
                    for k in range(self.K)
                ]
                feat_rows.append(np.append(s_l, loss_T))

                # apply aggregate: credit all in buffer at this agg index
                for k in buffer:
                    fake_last[k] = fake_agg
                buffer.clear()
                fake_agg += 1

            fake_slot += 1

        if feat_rows:
            preds = self.rgrs.predict(np.asarray(feat_rows, dtype=np.float32))
            util = float(np.sum(preds))
        return util
    
    def start_new_window(self, T):
        self.n_max = max(1, min(self.n_max, len(self._nonempty_slots)))
        self.n_min = max(1, min(self.n_min, self.n_max))
        self.current_a = self._pick_pattern(T, self.last_up.copy())
        self.pattern_idx = 0
        self.pattern_history.append(self.current_a.copy())

    def _connected_in_fake_slot(self, r):
        table_len = len(self._connectivity_table)
        return self._connectivity_table.get(r % table_len, [])

    def _current_loss_placeholder(self):
        """Replace with live val-loss."""
        return 1.0
    
    def is_window_start(self):
        return (self.pattern_idx % self.I0) == 0