from infill import infill
from screen import OUT, load_model, score_panel, write


def main():
    models = {n:load_model(n) for n in ("host","donor")}
    first, items = score_panel(models,"bits","discovery",infill("discovery"))
    second = None
    if first["difference"]>=.05:
        second, records = score_panel(models,"bits","confirmation",infill("confirmation"))
        items.extend(records)
    ok = bool(second and second["difference"]>0 and second["p"]<.05)
    write(OUT/"infill-screen.json", {"discovery":first,"confirmation":second,"qualified":ok})
    write(OUT/"infill-screen-items.json",items)


if __name__ == "__main__":
    main()
