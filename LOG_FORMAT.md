# Log format (shared by Half A and Half B)

One row = one event. One session = all rows with the same `session_id`.

## Columns
| column     | meaning                                              | example        |
|------------|------------------------------------------------------|----------------|
| session_id | unique id per solve                                  | h_001, b_042   |
| label      | human / bot_naive / bot_evasive                      | human          |
| t_ms       | milliseconds since the CAPTCHA appeared (int)        | 1450           |
| event      | mouse_move / click / key_down / key_up / submit      | key_down       |
| x, y       | mouse position in px (empty for key and submit rows) | 182, 145       |
| key        | typed character or Backspace (key rows only)         | K              |

## Rules
- t_ms starts at 0 when the CAPTCHA is shown.
- x, y are relative to the top-left corner of the CAPTCHA box, not the screen.
- The first row of each session is a mouse_move with the starting mouse position.
- The last row of each session is submit.
- mouse_move: browsers fire it about every 8–16 ms; bots should log at ~16 ms intervals.
- Every key press has a key_down row and a matching key_up row (same key).
- click is recorded on mouse button press, with the x, y where it happened.

## Page layout (px, inside a 400 × 250 box)
| element       | x range   | y range   |
|---------------|-----------|-----------|
| CAPTCHA image | 120 – 280 | 40 – 100  |
| input box     | 120 – 280 | 130 – 160 |
| submit button | 150 – 250 | 180 – 210 |

## Example session (shortened)
session_id,label,t_ms,event,x,y,key
h_001,human,0,mouse_move,30,220,
h_001,human,16,mouse_move,34,216,
...
h_001,human,1820,click,175,146,
h_001,human,2310,key_down,,,K
h_001,human,2405,key_up,,,K
...
h_001,human,5120,click,201,195,
h_001,human,5121,submit,,,

## Files
- Human logs: half_b_catcher/data/human_sessions.csv
- Bot logs:   half_a_breaker/logs/bot_sessions.csv