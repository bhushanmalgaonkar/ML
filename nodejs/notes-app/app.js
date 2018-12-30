const fs = require('fs');
const _ = require('lodash');
const yargs = require('yargs');

const notes = require('./notes');

titleOptions = {
    describe: 'Title of the note to read',
    demand: true,
    alias: 't'
}
bodyOptions = {
    describe: 'Actual content of the note',
    demand: true, 
    alias: 'b'
}

const argv = yargs
    .command('add', 'Add a new note', {
        title: titleOptions, 
        body: bodyOptions
    })
    .command('list', 'List all the notes')
    .command('read', 'Read a note', {
        title: titleOptions
    })
    .command('remove', 'Delete a note', {
        title: titleOptions
    })
    .help()
    .argv;
const command = argv._[0];

if (command === undefined) {
    console.error('Use one of the commands: add, list, read, remove');
} else if (command.toLowerCase() === 'add') {
    var note = notes.add(argv.title, argv.body);
    if (note) {
        console.log('Note created');
        console.log('--');
        console.log(`Title: ${note.title}`);
        console.log(`Body : ${note.body}`);
    } else {
        console.error('Note title already exists.')
    }
} else if (command.toLowerCase() === 'list') {
    var notesRead = notes.list();
    if (notesRead.length > 0) {
        notesRead.forEach(note => {
            console.log(note.title);
        });
    } else {
        console.log('Couldn\'d find any notes');
    }
} else if (command.toLowerCase() === 'read') {
    var read = notes.read(argv.title);
    if (read) {
        console.log('Note read');
        console.log('--');
        console.log(`Title: ${read.title}`);
        console.log(`Body : ${read.body}`);
    } else {
        console.log('Couldn\'t find note');
    }
} else if (command.toLowerCase() === 'remove') {
    var removed = notes.remove(argv.title);
    if (removed) {
        console.log(`Note removed`)
    } else {
        console.log(`No such note exists.`);
    }
} else {
    console.error(`Invalid command ${command}`);
}