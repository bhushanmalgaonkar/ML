const data_url = 'notes-data.json';
const fs = require('fs');

var fetchNotes = () => {
    try {
        return JSON.parse(fs.readFileSync(data_url));
    } catch (e) {
        console.error('Error occurred reading notes file');
        return []
    }
}

var saveNotes = (notes) => {
    fs.writeFileSync(data_url, JSON.stringify(notes));
}

var add = (title, body) => {
    var notes = fetchNotes();
    var duplicates = notes.filter((note) => note.title === title);
    if (duplicates.length === 0) {
        var note = { title, body };
        notes.push(note);

        saveNotes(notes);
        return note;
    }
}

var list = () => {
    return fetchNotes();
}

var read = (title) => {
    var notes = fetchNotes();
    var filtered = notes.filter((note) => note.title === title);
    if (filtered.length === 1)
        return filtered[0];
}

var remove = (title) => {
    var notes = fetchNotes();
    var filtered = notes.filter((note) => note.title !== title);
    saveNotes(filtered);
    return notes.length !== filtered.length;
}



module.exports = {
    add,
    list,
    read,
    remove
};