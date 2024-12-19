#!/bin/bash

# print out message about formatting
echo "Note: For ALL of the following prompts, do NOT use quotes when providing \
inputs. For example, \"stat214/lab1/lab1.ipynb\" will give an error, but \
stat214/lab1/lab1.ipynb will produce the intended result."

# get notebook path
echo "Enter the path to your notebook file."
read path

# get title for report
echo "Enter the desired title (header) of the report."
read title

authors=() # initialize an empty array to store inputs
author_counter=1 # initialize counter to properly number the authors

# prompt for input author until the user indicates they are done
while true; do

    # process each number differently for grammatical correctness
    case $author_counter in
        # for the first three authors, we put the word
        1)
            num="first"
            ;;
        2)
            num="second"
            ;;
        3)
            num="third"
            ;;
        # the proper suffix is 'th' for 4th through 20th
        # after 20th, it changes (e.g. 21st, 22nd) so we switch to #21, etc.
        *)
            # print out new message for proper grammar if it is #XX
            if [ "$author_counter" -gt 20 ]; then
                echo "Enter the name of author #${author_counter} \
                (or type \"none\" to finish)."
            else
                num="${author_counter}th"
            fi
            ;;
    esac

    # unless we already printed out the message, prompt for the XXth author
    if [ "$author_counter" -lt 21 ]; then
        echo "Enter the name of the ${num} author (or type \"none\" to finish)."
    fi

    # read input author
    read author
    
    # check if the user wants to exit the loop
    if [ "$author" = "none" ]; then
        break
    fi
    
    # add the author to the array
    authors+=("$author")
    ((author_counter++))

done

# get filename to write to
echo "What do you want the file name to be (must end in .pdf)?"
read output_name

# create temporary file
temp_path="${path%.ipynb}-temp.ipynb"

# add title and author metadata
python3 compile_helper.py add_metadata "$path" "$title" "${authors[@]}"

# use the following to convert to pdf by re-running the file
# jupyter nbconvert --execute --no-input --to pdf "$temp_path" \
#                   --output "$output_name"

# use the following to convert to pdf without re-running the file
# jupyter nbconvert --no-input --to pdf "$temp_path" --output "$output_name"

# use the following to convert to pdf including code
jupyter nbconvert --execute --to pdf "$temp_path" --output "$output_name"

# delete temporary file
python3 compile_helper.py delete_temp "$path"
