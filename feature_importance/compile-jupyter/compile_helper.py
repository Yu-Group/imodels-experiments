import nbformat
import os
import sys

def combine_names(authors: list[str]) -> str:
    """
    Combines names properly for the author section of the header.
    
    :param authors: A list where each element is the name of one author.
    """
    
    if len(authors) == 0:
        return ""
    elif len(authors) == 1:
        return authors[0]
    elif len(authors) == 2:
        return authors[0] + " and " + authors[1]
    else:
        return ", ".join(authors[:-1]) + ", and " + authors[-1]

def add_metadata(path: str, title: str, authors: list[str]) -> None:
    """
    Adds the title and author metadata to a temporary jupyter notebook file.
    
    :param path: The file path to the ipynb to be converted to pdf.
    :param title: The title to go in the header of the compiled document.
    :param authors: A list where each element is the name of one author.
    :raises TypeError: If the inputs are not of the correct type.
    :raises ValueError: If the path is badly formatted.
    """

    # manual type-checking
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path).__name__}.")
    if not isinstance(title, str):
        raise TypeError(f"title must be a string, got {type(title).__name__}.")
    if not isinstance(authors, list):
        raise TypeError((f"authors must be a list, "
                         f"got {type(authors).__name__}."))
    if not all(isinstance(author, str) for author in authors):
        raise TypeError("all elements in 'authors' must be strings, "\
            "but at least one was not.")
    if not path.endswith(".ipynb"):
        raise ValueError("the given path is not to a jupyter notebook.")

    # read in the notebook
    with open(path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # update the title and the authors
    notebook.metadata['title'] = title
    notebook.metadata['authors'] = [{"name": combine_names(authors)}]

    # re-save the notebook with the new metadata attached
    newpath = path[:-6] + "-temp.ipynb"
    with open(newpath, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
        
    return

def delete_temp(path: str) -> None:
    """
    Deletes the temporary file at the specified path.
    
    :param path: The name of the jupyter notebook whose temporary version
                 we would like to delete.
    :raises FileNotFoundError: If the file does not exist.
    :raises PermissionError: If there are insufficient
                             permissions to delete the file.
    :raises Exception: For any other issues that occur during file deletion.
    """
    
    # check that the path is valid
    if not isinstance(path, str):
        raise TypeError(f"path must be an string, got {type(path).__name__}.")
    if not path.endswith(".ipynb"):
        raise ValueError("the given path is not to a jupyter notebook.")
    
    # get the path of the temporary file we want to delete
    newpath = path[:-6] + "-temp.ipynb"
    try:
        os.remove(newpath)
    except FileNotFoundError:
        print(f"Error: The file '{newpath}' does not exist.")
    except PermissionError:
        print(f"Error: Insufficient permissions to delete '{newpath}'.")
    except Exception as e:
        print(f"An error occurred while trying to delete '{newpath}': {e}") 
    
    return

if __name__ == "__main__":
    
    function_name = sys.argv[1]
    if function_name == "add_metadata":
        add_metadata(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif function_name == "delete_temp":
        delete_temp(sys.argv[2])
    else:
        raise ValueError("invalid function name.")