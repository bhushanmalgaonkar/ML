const yargs = require('yargs');
const geocode = require('./modules/geocode');
const weather = require('./modules/weather');


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


geocode.geocodeAddress(argv.address, (errorMessage, geoResult) => {
    if (errorMessage) {
        console.log(errorMessage);
    } else {
        weather.getWeather(geoResult.lat, geoResult.lng, (errorMessage, weatherResult) => {
            if (errorMessage) {
                console.log(errorMessage);
            } else {
		console.log(geoResult.address);
                console.log(`It is ${weatherResult.temperature}f. It feels like ${weatherResult.apparentTemperature}f`);
            }
        });
    }
});


