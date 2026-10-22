# poker_sandbox — адаптер ботов для изолированного стенда

Абстрактный контракт `PokerAdapter`, через который боты работают со стендом, и
офлайн-реализация `MockAdapter` (NL Texas Hold'em, heads-up … 6-max) для локальных
прогонов red-team сценариев против собственных анти-чит детекторов. Без сети и
без реальных денег.

```
adapter/
  base.py     PokerAdapter (ABC) + исключения
  models.py   Card, TableState, PlayerState, Action, ActionRecord, Pot, HandResult, Event
  config.py   AdapterConfig / Stakes / RateLimits (pydantic v2)
  mock.py     MockAdapter: движок NLHE + оценщик рук (evaluate_hand)
  audit.py    JsonlAuditLog: запись всех событий в JSONL
tests/test_mock.py
```

## Запуск

```bash
cd poker_sandbox
pip install -e '.[dev]'   # или: pip install 'pydantic>=2.5' pytest
pytest -q
```

## Пример

```python
import random
from adapter import Action, ActionType, AdapterConfig, EventType, RateLimits
from adapter.mock import MockAdapter

cfg = AdapterConfig(
    seats=6, seed=42, hero_seat=0,
    limits=RateLimits(max_hands=20),
    log_path="logs/session.jsonl",
)
rng = random.Random(0)
adapter = MockAdapter()
with adapter.session(cfg):
    while (st := adapter.view(None)).to_act_seat is not None:   # локально играем за все места
        kind = rng.choice(st.legal_actions)
        amount = rng.randint(st.min_raise_to, st.max_raise_to) if kind in (ActionType.BET, ActionType.RAISE) else 0
        adapter.send_action(Action(st.to_act_seat, kind, amount, hand_id=st.hand_id))
    last = adapter.get_hand_history(adapter.hand_ids[-1])
    print(len(adapter.hand_ids), "hands; last:", last.net, last.hand_ranks)
```

Событийный стиль (бот реагирует на `TURN`; вызывать `send_action` из колбэка можно):

```python
def bot(seat):
    def on_event(e):
        if e.type is EventType.TURN and e.payload["seat"] == seat:
            kind = ActionType.CHECK if ActionType.CHECK in e.payload["legal_actions"] else ActionType.CALL
            adapter.send_action(Action(seat, kind, hand_id=e.hand_id))
    return on_event

for s in range(cfg.seats):
    adapter.on_event(bot(s), seat=s)    # seat= — чьи приватные события (карманные карты) получать
adapter.connect(cfg)                    # при auto_deal=True раздачи идут до max_hands
```

## Ключевые решения

- **Суммы.** Для `BET`/`RAISE` `amount` — итоговая ставка на улице («raise to»).
  `ALL_IN` движок нормализует в `BET`/`CALL`/`RAISE`. `TableState` содержит
  `legal_actions`, `to_call`, `min_raise_to`, `max_raise_to` для того, кто ходит.
- **Видимость.** `Event.visibility`: `PUBLIC` — всем; `PRIVATE` — только
  подписчику места `recipient_seat` (карманные карты); `AUDIT` — никогда не
  доставляется ботам, только в лог (seed, полная колода, ручные правки стеков).
  `get_table_state()` показывает карты только `hero_seat`; `MockAdapter.view(seat)`
  — вид от лица любого места.
- **Воспроизводимость.** Одинаковый `seed` + одинаковые действия → побайтно
  одинаковый лог (при инжектированном `clock`). Если `seed=None`, он выбирается
  случайно и пишется в `session_started`.
- **Аудит.** Каждая строка JSONL — `Event` (`type`, `timestamp`, `seq`, `hand_id`,
  `payload`, `visibility`). Логируются и принятые (`action`), и отклонённые
  (`action_rejected` с причиной) действия. Токен хранится как `SecretStr` и в лог
  попадает замаскированным.
- **Правила.** Мин. бет = BB, мин. рейз = размер последнего полного рейза; олл-ин
  меньше полного рейза не переоткрывает торговлю уже сходившим; сайд-поты, возврат
  неуравненной ставки, сплит с нечётной фишкой первому слева от баттона, раннаут
  при олл-инах.
  Упрощения: без анте/страддлов/dead button/burn-карт, на шоудауне открываются все.
- **Тест-хуки мока:** `start_hand(hole_cards=..., board=..., button=...)` для
  заряженной колоды и `set_stack(seat, chips)` между раздачами (`auto_deal=False`).

## Подключение реального API

Реализуйте `PokerAdapter` (например, `adapter/http.py`): `connect` открывает
сессию по `base_url`/`api_token`, `on_event` транслирует события сервера в
`Event`, `send_action` соблюдает `limits.max_actions_per_second` и
`limits.action_timeout_s`, а для аудита используйте тот же `JsonlAuditLog`.
Логика ботов, завязанная только на `PokerAdapter` и модели, не меняется.
Контракт синхронный и однопоточный; асинхронный вариант можно добавить
отдельным `AsyncPokerAdapter` с теми же моделями.
