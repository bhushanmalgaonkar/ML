const fs = require('fs');
const request = require('request');

var apikey = JSON.parse(fs.readFileSync('apikey', 'utf8'));

var getWeather = (lat, lng, callback) => {
    request({
        uri: `https://api.darksky.net/forecast/${apikey['darksky']}/${lat},${lng}`,
        json: true
    }, (error, response, body) => {
        if (error) {
            callback('Unable to connect to darksky.net', undefined);
        } else if (response.statusCode !== 200) {
            callback('Error while fetching weather', undefined);
        } else {
            callback(undefined, 
                { 
                    temperature: body.currently.temperature,
                    apparentTemperature: body.currently.apparentTemperature
                }
            );
        }
    });
};


module.exports = {
    getWeather
};