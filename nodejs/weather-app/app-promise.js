const fs = require('fs');
const yargs = require('yargs');
const axios = require('axios');


var argv = yargs
    .options({
        a: {
            demand: true,
            alias: 'address',
            describe: 'Address to fetch weather for',
            string: true
        }
    })
    .help()
    .alias('help', 'h')
    .argv;


var encodedAddress = encodeURIComponent(argv.address);
var apikey = JSON.parse(fs.readFileSync('apikey', 'utf8'));

var geocodeURI = `https://maps.googleapis.com/maps/api/geocode/json?address=${encodedAddress}&key=${apikey['maps']}`;
axios.get(geocodeURI).then((response) => {
    if (response.data.status !== 'OK')
        throw new Error(response.data.error_message);
    if (response.data.stats == 'ZERO_RESULTS')
        throw new Error('Couldn\'t find the address');
    
    var address = response.data.results[0].formatted_address;
    console.log(address);

    var lat = response.data.results[0].geometry.location.lat;
    var lng = response.data.results[0].geometry.location.lng;
    
    var weatherURI = `https://api.darksky.net/forecast/${apikey['darksky']}/${lat},${lng}`;
    return axios.get(weatherURI);
}).then((response) => {
    var temperature = response.data.currently.temperature;
    var apparentTemperature = response.data.currently.apparentTemperature;

    console.log(`It\'s currently ${temperature}f. It feels like ${apparentTemperature}f`);
}).catch((e) => {
    if (e.code === 'ENOTFOUND') {
        console.log('Host not found.');
    } else {
        console.log(e.message);
    }
});
