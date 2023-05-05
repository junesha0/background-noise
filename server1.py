from typing import Dict, List, Optional
from flask import Flask, request, jsonify
import pathlib
import uuid
import json


app = Flask(__name__)
thisdir = pathlib.Path(__file__).parent.absolute() # path to directory of this file


# Function to load and save the mail to/from the json file
def load_mail() -> List[Dict[str, str]]:
    """
    Loads the mail from the json file
    Returns:
        list: A list of dictionaries representing the mail entries
    """
    try:
        return json.loads(thisdir.joinpath('mail_db.json').read_text())
    except FileNotFoundError:
        return []


def save_mail(mail: List[Dict[str, str]]) -> None:
    """
    Converts mail into a json string and writes the string to 'mail_db.json' in the current directory
    
    Args:
    	mail (List[Dict[str, str]]): A list of dictionaries representing the mail entries
    Returns:
       Nothing.
    """
    thisdir.joinpath('mail_db.json').write_text(json.dumps(mail, indent=4))


def add_mail(mail_entry: Dict[str, str]) -> str:
    """
    Adds the mail entry into mail inbox, by appending it to the end of the list of dictionaries. Then, generates a unique ID for that mail entry.
    It then saves the new mail inbox with the added mail. Then, returns the mail entry id.
    
    Args:
    	mail_entry (Dict[str, str]): a dictionary representing a mail entry
    	
    Returns:
    	str: the id of the mail entry
    """
    mail = load_mail()
    mail.append(mail_entry)
    mail_entry['id'] = str(uuid.uuid4()) # generate a unique id for the mail entry
    save_mail(mail)
    return mail_entry['id']


def delete_mail(mail_id: str) -> bool:
    """
    If the mail ID exists, delete the mail and then save the mail. Then return TRUE for having deleted the mail. If the mail ID does not exist,
    If the mail ID does not exist, just return FALSE
    
    Args:
    	mail_id (str): the id of the mail entry
    	
    Returns:
    	bool: True for success, false otherwise
    """
    mail = load_mail()
    for i, entry in enumerate(mail):
        if entry['id'] == mail_id:
            mail.pop(i)
            save_mail(mail)
            return True

    return False


def get_mail(mail_id: str) -> Optional[Dict[str, str]]:
    """
    Gets a mail entry from the json file:
    Checks all the dictionaries (mails) in the list of mails; checks the definition listed under all "id"s, and checks if it matches the mail id
    If it does, return the mail entry dictionary. If there is no mail matching the id, return nothing
    
    Args:
    	mail_id (str): the id of the mail entry
    
    Returns:
    	(Optional) Dict: A Dictionary representing the mail entry if found. If entry doesn't exist, returns None
    """
    mail = load_mail()
    for entry in mail:
        if entry['id'] == mail_id:
            return entry

    return None


def get_inbox(recipient: str) -> List[Dict[str, str]]:
    """
    Gets all entry to a specific recipient from all mail, by checking all dictionaries in list for the definition under "Recipient"
    
    Args:
    	recipient (str): the recipient of the mail
    	
    Returns:
    	list: A list of dictionaries representing the recepient's inbox
    """
    mail = load_mail()
    inbox = []
    for entry in mail:
        if entry['recipient'] == recipient:
            inbox.append(entry)

    return inbox


def get_sent(sender: str) -> List[Dict[str, str]]:
    """
    Gets all entry from a specific sender from all mail, by checking all dictionaries in list for the definition under "Sender"
    
    Args:
    	sender (str): the sender of the mail
    	
    Returns:
    	list: A list of dictionaries representing the sent mail
    """
    mail = load_mail()
    sent = []
    for entry in mail:
        if entry['sender'] == sender:
            sent.append(entry)

    return sent


# API routes - these are the endpoints that the client can use to interact with the server
@app.route('/mail', methods=['POST'])
def add_mail_route():
    """
    Summary: Adds a new mail entry to the json file
    Returns:
        str: The id of the new mail entry
    """
    mail_entry = request.get_json() # mail entry is a dictionary
    mail_id = add_mail(mail_entry) 
    res = jsonify({'id': mail_id})
    res.status_code = 201 # Status code for "created"
    return res


@app.route('/mail/<mail_id>', methods=['DELETE'])
def delete_mail_route(mail_id: str):
    """
    Summary: Deletes a mail entry from the json file
    Args:
        mail_id (str): The id of the mail entry to delete
    Returns:
        bool: True if the mail was deleted, False otherwise
    """
    # TODO: implement this function
    if delete_mail(mail_id) == True:
    	res = jsonify("Deleted")
    	res.status_code = 200
    else:
    	res = jsonify("Not Found")
    	res.status_code = 404
    return res


@app.route('/mail/<mail_id>', methods=['GET'])
def get_mail_route(mail_id: str):
    """
    Summary: Gets a mail entry from the json file
    Args:
        mail_id (str): The id of the mail entry to get
    Returns:
        dict: A dictionary representing the mail entry if it exists, None otherwise
    """
    res = jsonify(get_mail(mail_id))
    res.status_code = 200 # Status code for "ok"
    return res


@app.route('/mail/inbox/<recipient>', methods=['GET'])
def get_inbox_route(recipient: str):
    """
    Summary: Gets all mail entries for a recipient from the json file
    Args:
        recipient (str): The recipient of the mail
    Returns:
        list: A list of dictionaries representing the mail entries
    """
    res = jsonify(get_inbox(recipient))
    res.status_code = 200
    return res


# TODO: implement a route to get all mail entries for a sender
# HINT: start with soemthing like this:
#   @app.route('/mail/sent/<sender>', ...)
@app.route('/mail/sent/<sender>', methods=['GET'])
def get_sent_route(sender: str):
	"""
	Summary: Gets all the messages sent by sender from the output of get_sent
	
	Args:
		sender(str): sender of the mail
		
	Returns:
		list of dictionaries of emails of sender in json syntax (since json is a syntax for java objects?)
		
	"""
	res = jsonify(get_sent(sender))
	res.status_code = 200
	return res
  

if __name__ == '__main__':
    app.run(port=5000, debug=True)