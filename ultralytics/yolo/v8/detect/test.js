function calculateDaysBetweenDates(begin, end) {
    const diff = Math.abs(begin - end);
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
    }
    