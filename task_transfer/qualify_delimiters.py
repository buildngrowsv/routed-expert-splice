from delimiters import delimiter_panel
from screen import OUT, load_model, score_panel, write


def main():
    models = {n:load_model(n) for n in ("host","donor")}
    first, items = score_panel(models,"delimiters","discovery",delimiter_panel("discovery"))
    second = None
    if first["difference"]>=.05:
        second, records = score_panel(models,"delimiters","confirmation",delimiter_panel("confirmation"))
        items.extend(records)
    ok = bool(second and second["difference"]>0 and second["p"]<.05)
    write(OUT/"delimiter-screen.json", {"discovery":first,"confirmation":second,"qualified":ok})
    write(OUT/"delimiter-screen-items.json",items)


if __name__ == "__main__":
    main()
