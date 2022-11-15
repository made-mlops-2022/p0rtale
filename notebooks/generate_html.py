import os
import click


@click.command(name="notebook_to_html")
@click.argument("notebook_path")
def convert_notebook_to_html_command(notebook_path: str):
    os.system('jupyter nbconvert --to html ' + notebook_path)


if __name__ == "__main__":
    convert_notebook_to_html_command()
